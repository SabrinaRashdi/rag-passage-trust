
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from compute_trust_scores import (
    SET_NEIGHBOURS, load_embeddings, reference_masks,
)
from evaluate_adaptive_attack import SAVED_FEATURES, passage_features
from evaluate_detection import (
    N_BOOTSTRAPS, SEED, TARGET_FPR, auroc, cross_validated_scores,
    tpr_at_fpr,
)
from evaluate_end_to_end import (
    READ_K, RETRIEVE_K, answer_matches, score_answers,
)
from evaluate_end_to_end_llm import MODEL_NAME, llm_answers
from generate_reverse_questions import (
    N_QUESTIONS, generate_questions, question_signals,
)
from generate_trust_embeddings import MODEL_NAME as CONTRIEVER_NAME
from generate_trust_embeddings import ENCODE_DEVICE, encode
from src.contriever_src.contriever import Contriever
from src.trust import set_density

# An attack aimed at the reverse-question defence ("diverse" attack). An
# attacker who knows the defence wants each poisoned passage to answer
# several questions. Each Kind 1 poisoned passage (question + "." + text)
# gets the first FILLER_WORDS words of a real passage on the same topic at
# the end: the j-th poisoned passage of a question takes the j-th
# best-ranked benign passage of that question that is not gold and does
# not contain the correct answer (repeated if there are fewer than 5). Detection uses all benign passages; the
# end-to-end test uses the Qwen reader (the extractive reader's support
# scores exist only for the original passages).
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"
DATASET_PATH = DATA_DIR / "nq_passage_dataset.json"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
REVERSE_SCORES_PATH = TRUST_DIR / "reverse_question_scores.csv"

PASSAGES_PATH = TRUST_DIR / "diverse_attack_passages.jsonl"
QUESTIONS_PATH = TRUST_DIR / "diverse_attack_questions.csv"
OUTPUT_PATH = TRUST_DIR / "diverse_attack_results.json"
# Features of the candidate sets under this attack, for
# evaluate_robustness.py
SCORES_PATH_OUT = TRUST_DIR / "diverse_attack_scores.csv"

BASE_ATTACK = "prefix"
ATTACK = "diverse"
FILLER_WORDS = 40
METHODS = (
    "retrieval_score", "set_density", "shortness", "question_focus",
    "question_sameness", "retrieval+focus",
)
END_TO_END_METHODS = (
    "retrieval_score", "set_density", "question_focus", "retrieval+focus",
)


def main():
    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = {
            p["passage_id"]: p for p in (json.loads(line) for line in f)
        }

    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["attack"] in ("none", BASE_ATTACK)]

    with REVERSE_SCORES_PATH.open(encoding="utf-8") as f:
        reverse = {r["passage_id"]: r for r in csv.DictReader(f)}

    retrieval = {r["passage_id"]: float(r["score_original"]) for r in rows}
    questions = sorted(dataset)

    # 1. The new poisoned passages.
    attacked = []

    for qid in questions:
        # The correct answer is checked as whole words, as in the
        # end-to-end test.
        fillers = sorted(
            (pid for pid, p in passages.items()
             if p["query_id"] == qid and p["label"] == 0
             and not p["is_gold"]
             and not answer_matches(p["text"],
                                    dataset[qid]["correct_answer"])),
            key=lambda pid: -retrieval[pid],
        )
        assert fillers, qid
        poisoned = sorted(
            pid for pid, p in passages.items()
            if p["query_id"] == qid and p["attack"] == BASE_ATTACK
        )

        for j, pid in enumerate(poisoned):
            filler = passages[fillers[j % len(fillers)]]["text"].split()
            attacked.append({
                "passage_id": pid.replace(BASE_ATTACK, ATTACK),
                "query_id": qid,
                "label": 1,
                "is_gold": False,
                "attack": ATTACK,
                "filler_from": fillers[j % len(fillers)],
                "text": passages[pid]["text"] + " "
                        + " ".join(filler[:FILLER_WORDS]),
            })

    with PASSAGES_PATH.open("w", encoding="utf-8") as f:
        for p in attacked:
            f.write(json.dumps(p) + "\n")

    # 2. Retrieval embeddings, reverse questions and their embeddings.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generated = generate_questions(attacked, QUESTIONS_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_NAME)
    model = Contriever.from_pretrained(CONTRIEVER_NAME).eval()
    passage_emb = encode([p["text"] for p in attacked], tokenizer, model,
                         ENCODE_DEVICE, "diverse passages")
    question_emb = encode(
        [q for p in attacked for q in generated[p["passage_id"]]],
        tokenizer, model, ENCODE_DEVICE, "diverse questions",
    )

    query_emb, query_index = load_embeddings("query")
    pool_emb, pool_index = load_embeddings("pool")

    # Reader, perplexity and consistency features, as for the reworded
    # attack.
    feature_rows, _ = passage_features(
        attacked, passage_emb, ATTACK, queries, query_emb, query_index,
        pool_emb, reference_masks(pool_index, passages.values()), device,
    )

    def user(qid):
        return query_emb[query_index[f"{qid}_original"]]

    new = {}

    for n, p in enumerate(attacked):
        focus, sameness = question_signals(
            question_emb[n * N_QUESTIONS:(n + 1) * N_QUESTIONS],
            user(p["query_id"]),
        )
        new[p["passage_id"]] = {
            "embedding": passage_emb[n],
            "retrieval_score": float(user(p["query_id"]) @ passage_emb[n]),
            "question_focus": focus,
            "question_sameness": sameness,
        }

    for p in attacked:
        passages[p["passage_id"]] = p

    # 3. Detection: all benign passages against the new poisoned ones.
    table = [r["passage_id"] for r in rows if r["attack"] == "none"]
    table += [p["passage_id"] for p in attacked]
    labels = np.array([passages[pid]["label"] for pid in table])
    groups = np.array([passages[pid]["query_id"] for pid in table])

    def embedding(pid):
        if pid in new:
            return new[pid]["embedding"]
        return pool_emb[pool_index[passages[pid]["doc_id"]]]

    def value(pid, name):
        if pid in new:
            return new[pid][name]
        if name == "retrieval_score":
            return retrieval[pid]
        return float(reverse[pid][name])

    features = {
        name: np.array([value(pid, name) for pid in table])
        for name in ("retrieval_score", "question_focus",
                     "question_sameness")
    }
    features["shortness"] = -np.array(
        [len(passages[pid]["text"].split()) for pid in table], dtype=float
    )
    features["set_density"] = np.zeros(len(table))

    for qid in questions:
        members = np.flatnonzero(groups == qid)
        features["set_density"][members] = set_density(
            np.stack([embedding(table[i]) for i in members]), SET_NEIGHBOURS
        )

    features["retrieval+focus"] = cross_validated_scores(
        np.stack([features["retrieval_score"], features["question_focus"]],
                 axis=1),
        labels, groups,
    )

    known_rows = {r["passage_id"]: r for r in rows}
    known_rows.update((r["passage_id"], r) for r in feature_rows)

    with SCORES_PATH_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "passage_id", "query_id", "label", *SAVED_FEATURES,
            "question_focus", "question_sameness", "set_density",
        ])

        for i, pid in enumerate(table):
            writer.writerow([
                pid, passages[pid]["query_id"], labels[i],
                *[known_rows[pid][name] for name in SAVED_FEATURES],
                features["question_focus"][i],
                features["question_sameness"][i],
                features["set_density"][i],
            ])

    rng = np.random.default_rng(SEED)
    rows_of = {q: np.flatnonzero(groups == q) for q in questions}
    samples = [
        np.concatenate([rows_of[q] for q in rng.choice(questions,
                                                       len(questions))])
        for _ in range(N_BOOTSTRAPS)
    ]
    results = {"detection": {}}

    with (TRUST_DIR / "reverse_question_results.json").open(
        encoding="utf-8"
    ) as f:
        known = json.load(f)[BASE_ATTACK]["methods"]

    for name in METHODS:
        s = features[name]
        low, high = np.percentile(
            [auroc(labels[i], s[i]) for i in samples], [2.5, 97.5]
        )
        results["detection"][name] = {
            "auroc": auroc(labels, s),
            "auroc_ci_95": [float(low), float(high)],
            "tpr_at_5pct_fpr": tpr_at_fpr(labels, s),
            "auroc_kind1": known[name]["auroc"],
        }

    # Does the attack still reach the reader?
    row_of = {pid: i for i, pid in enumerate(table)}
    candidates = {}

    for condition in ("clean", "attacked"):
        for qid in questions:
            pool = [
                pid for pid in table
                if passages[pid]["query_id"] == qid and (
                    (passages[pid]["label"] == 0
                     and passages[pid]["retrieved_top10"])
                    or (condition == "attacked" and pid in new)
                )
            ]
            pool.sort(key=lambda pid: -features["retrieval_score"][
                row_of[pid]])
            candidates[condition, qid] = pool[:RETRIEVE_K]

    results["poisoned_in_top10"] = float(np.mean([
        sum(pid in new for pid in candidates["attacked", q])
        for q in questions
    ]))

    # 4. End-to-end with the LLM reader, same rules as
    # evaluate_end_to_end.py (thresholds flag 5% of clean top-10 passages,
    # set density inside each retrieved list).
    density = {}

    for key, pids in candidates.items():
        density[key] = dict(zip(pids, set_density(
            np.stack([embedding(pid) for pid in pids]), SET_NEIGHBOURS
        )))

    def suspicion(method, condition, qid, pid):
        if method == "set_density":
            return density[condition, qid][pid]
        return features[method][row_of[pid]]

    runs = {ATTACK: {}}

    for method in ("no_defence", *END_TO_END_METHODS):
        threshold = None

        if method != "no_defence":
            threshold = float(np.quantile([
                suspicion(method, "clean", q, pid)
                for q in questions for pid in candidates["clean", q]
            ], 1 - TARGET_FPR))

        run = {"threshold": threshold}

        for condition in ("clean", "attacked"):
            run[condition] = {}

            for qid in questions:
                pids = candidates[condition, qid]
                flagged = set() if threshold is None else {
                    pid for pid in pids
                    if suspicion(method, condition, qid, pid) > threshold
                }
                kept = [pid for pid in pids if pid not in flagged]
                run[condition][qid] = (flagged, kept[:READ_K])

        runs[ATTACK][method] = run

    answer_of = llm_answers(dataset, passages, runs)
    results["end_to_end"] = score_answers(
        dataset, passages, questions, runs, answer_of
    )[ATTACK]

    with (TRUST_DIR / "reverse_end_to_end_results.json").open(
        encoding="utf-8"
    ) as f:
        known_success = json.load(f)[MODEL_NAME][BASE_ATTACK]

    for method, outcome in results["end_to_end"].items():
        outcome["attack_success_kind1"] = (
            known_success[method]["attacked"]["attack_success"]["rate"]
        )

    results["setup"] = {
        "base_attack": BASE_ATTACK,
        "filler_words": FILLER_WORDS,
        "reader": MODEL_NAME,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDIVERSE ATTACK (Kind 1 poison + 40 words of a real passage)")
    print("=" * 70)
    print(f"{'method':<22}{'AUROC Kind 1':>14}{'AUROC diverse':>15}"
          f"{'95% CI':>18}{'TPR@5%':>9}")

    for name, d in results["detection"].items():
        print(f"{name:<22}{d['auroc_kind1']:>14.3f}{d['auroc']:>15.3f}"
              f"   [{d['auroc_ci_95'][0]:.3f}, {d['auroc_ci_95'][1]:.3f}]"
              f"{d['tpr_at_5pct_fpr']:>9.3f}")

    print(f"\nPoisoned passages in the top 10: "
          f"{results['poisoned_in_top10']:.2f} of 5")
    print(f"\nEnd-to-end with {MODEL_NAME}")
    print(f"{'method':<22}{'success Kind 1':>16}{'success diverse':>17}"
          f"{'poison read':>13}{'clean correct':>15}{'gold removed':>14}")

    for method, o in results["end_to_end"].items():
        a, c = o["attacked"], o["clean"]
        print(f"{method:<22}{o['attack_success_kind1']:>16.2f}"
              f"{a['attack_success']['rate']:>17.2f}"
              f"{a['mean_poison_read']:>13.2f}"
              f"{c['correct']['rate']:>15.2f}{c['gold_removed']:>14d}")

    print("Features saved to:", SCORES_PATH_OUT)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
