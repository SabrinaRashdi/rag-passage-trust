
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import SET_NEIGHBOURS, load_embeddings
from evaluate_detection import (
    ATTACKS, COMBINATIONS, N_BOOTSTRAPS, SEED, TARGET_FPR,
    cross_validated_scores,
)
from src.trust import normalize_answer, set_density

# End-to-end effect of each defence on the answers of a small RAG system.
# For every question: retrieve the top 10 passages for the original
# question, remove the passages the defence flags, give the 5 best-ranked
# remaining passages to the extractive reader, and take the answer with the
# highest reader support. We count how often the answer is the attacker's
# target answer (attack success rate, as in PoisonedRAG) and how often it
# is the correct answer, with and without poisoned passages in the corpus.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

DATASET_PATH = DATA_DIR / "nq_passage_dataset.json"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
ANSWER_SUPPORT_PATH = TRUST_DIR / "answer_support.csv"
OUTPUT_PATH = TRUST_DIR / "end_to_end_results.json"

RETRIEVE_K = 10
READ_K = 5

# Per-passage defences: column in trust_scores.csv and its sign
# (-1 when a smaller value is the more suspicious one).
PASSAGE_METHODS = {
    "retrieval_z": ("z_original", 1),
    "shortness": ("num_words", -1),
    "perplexity": ("log_perplexity", 1),
    "reader_support": ("support_original", 1),
    "z_drop_k3": ("z_drop", 1),
}
COMBINED = "combined_consistency"
# Combined features, as in evaluate_detection.py.
COMBINED_FEATURES = ("z_drop", "support_drop", "answer_disagreement")
assert len(COMBINED_FEATURES) == len(COMBINATIONS[COMBINED])


def answer_matches(prediction, target):
    """True if one answer contains the other as a run of whole words
    (after SQuAD-style normalisation)."""
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(target).split()

    if not pred or not gold:
        return False

    def contains(longer, shorter):
        return any(
            longer[i:i + len(shorter)] == shorter
            for i in range(len(longer) - len(shorter) + 1)
        )

    return contains(pred, gold) or contains(gold, pred)


def main():
    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = {
            p["passage_id"]: p for p in (json.loads(line) for line in f)
        }

    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with ANSWER_SUPPORT_PATH.open(encoding="utf-8") as f:
        reader = {row["passage_id"]: row for row in csv.DictReader(f)}

    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")

    def embedding(pid):
        p = passages[pid]
        return (adv_emb[adv_index[pid]] if p["label"] == 1
                else pool_emb[pool_index[p["doc_id"]]])

    row_of = {r["passage_id"]: i for i, r in enumerate(rows)}
    attacks = np.array([r["attack"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    retrieval = np.array([float(r["score_original"]) for r in rows])

    passage_scores = {
        name: sign * np.array([float(r[column]) for r in rows])
        for name, (column, sign) in PASSAGE_METHODS.items()
    }
    combined_features = np.stack([
        np.array([float(r[name]) for r in rows]) for name in COMBINED_FEATURES
    ], axis=1)

    questions = sorted(dataset)
    rng = np.random.default_rng(SEED)
    bootstrap = [
        rng.integers(0, len(questions), len(questions))
        for _ in range(N_BOOTSTRAPS)
    ]

    def rate(values):
        values = np.array(values, dtype=float)
        low, high = np.percentile(
            [values[sample].mean() for sample in bootstrap], [2.5, 97.5]
        )
        return {"rate": float(values.mean()),
                "ci_95": [float(low), float(high)]}

    results = {}

    for attack in ATTACKS:
        in_attack = (attacks == "none") | (attacks == attack)
        scores = dict(passage_scores)

        # The combined defence is trained on other questions only.
        combined = np.full(len(rows), np.nan)
        combined[in_attack] = cross_validated_scores(
            combined_features[in_attack], labels[in_attack], groups[in_attack]
        )
        scores[COMBINED] = combined

        # Candidate lists: the top 10 retrieved passages, with and without
        # this question's 5 poisoned passages in the corpus. The clean pool
        # ranking is the PoisonedRAG top 10, so only these can enter it.
        candidates = {}

        for condition in ("clean", "attacked"):
            for qid in questions:
                pool = [
                    pid for pid, p in passages.items()
                    if p["query_id"] == qid and (
                        (p["attack"] == "none" and p["retrieved_top10"])
                        or (condition == "attacked" and p["attack"] == attack)
                    )
                ]
                pool.sort(key=lambda pid: -retrieval[row_of[pid]])
                candidates[condition, qid] = pool[:RETRIEVE_K]

        # Set density is computed inside each retrieved list.
        density = {}

        for (condition, qid), pids in candidates.items():
            values = set_density(
                np.stack([embedding(pid) for pid in pids]), SET_NEIGHBOURS
            )
            density[condition, qid] = dict(zip(pids, values))

        def suspicion(method, condition, qid, pid):
            if method == "set_density":
                return density[condition, qid][pid]
            return scores[method][row_of[pid]]

        methods = ["no_defence", "remove_all_poison", *scores, "set_density"]
        results[attack] = {}

        for method in methods:
            # Threshold: flag 5% of the passages in clean retrieval lists.
            if method in ("no_defence", "remove_all_poison"):
                threshold = None
            else:
                clean_values = [
                    suspicion(method, "clean", qid, pid)
                    for qid in questions
                    for pid in candidates["clean", qid]
                ]
                threshold = float(np.quantile(clean_values, 1 - TARGET_FPR))

            outcome = {}

            for condition in ("clean", "attacked"):
                success, correct, removed = [], [], []
                poison_read, gold_removed = [], []

                for qid in questions:
                    record = dataset[qid]
                    pids = candidates[condition, qid]

                    if method == "no_defence":
                        flagged = set()
                    elif method == "remove_all_poison":
                        flagged = {
                            pid for pid in pids
                            if passages[pid]["label"] == 1
                        }
                    else:
                        flagged = {
                            pid for pid in pids
                            if suspicion(method, condition, qid, pid)
                            > threshold
                        }

                    kept = [pid for pid in pids if pid not in flagged]
                    read = kept[:READ_K]

                    if read:
                        best = max(
                            read,
                            key=lambda pid: float(
                                reader[pid]["support_original"]
                            ),
                        )
                        answer = reader[best]["answer_original"]
                    else:
                        answer = ""

                    success.append(
                        answer_matches(answer, record["incorrect_answer"])
                    )
                    correct.append(
                        answer_matches(answer, record["correct_answer"])
                    )
                    removed.append(len(flagged))
                    poison_read.append(
                        sum(passages[pid]["label"] == 1 for pid in read)
                    )
                    gold_removed.append(
                        sum(passages[pid]["is_gold"] for pid in flagged)
                    )

                outcome[condition] = {
                    "attack_success": rate(success),
                    "correct": rate(correct),
                    "mean_removed": float(np.mean(removed)),
                    "mean_poison_read": float(np.mean(poison_read)),
                    "gold_removed": int(np.sum(gold_removed)),
                }

            outcome["threshold"] = threshold
            results[attack][method] = outcome

    results["setup"] = {
        "questions": len(questions),
        "retrieve_k": RETRIEVE_K,
        "read_k": READ_K,
        "target_fpr": TARGET_FPR,
        "gold_in_clean_top10": int(sum(
            passages[pid]["is_gold"]
            for qid in questions for pid in candidates["clean", qid]
        )),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEND-TO-END EFFECT OF EACH DEFENCE")
    print(f"(top {RETRIEVE_K} retrieved, flagged passages removed, top "
          f"{READ_K} read; threshold flags {TARGET_FPR:.0%} of clean "
          f"passages)")
    print("=" * 84)

    for attack in ATTACKS:
        print(f"\nAttack: {attack}")
        print(f"{'method':<22}{'attack success':>16}{'correct':>10}"
              f"{'poison read':>13}{'clean correct':>15}{'gold removed':>14}")

        for method, o in results[attack].items():
            a, c = o["attacked"], o["clean"]
            print(f"{method:<22}{a['attack_success']['rate']:>16.2f}"
                  f"{a['correct']['rate']:>10.2f}"
                  f"{a['mean_poison_read']:>13.2f}"
                  f"{c['correct']['rate']:>15.2f}"
                  f"{c['gold_removed']:>14d}")

    print("\nGold passages in clean top 10:",
          results["setup"]["gold_in_clean_top10"])
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
