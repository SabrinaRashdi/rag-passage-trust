
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import SET_NEIGHBOURS, load_embeddings
from evaluate_detection import (
    ATTACKS, SEED, TARGET_FPR, auroc, cross_validated_scores,
)
from evaluate_poison_count import N_DRAWS, POISON_COUNTS
from generate_reverse_questions import (
    EMBEDDING_NAME, N_QUESTIONS, QUESTIONS_PATH, SCORES_PATH,
    question_signals,
)
from src.trust import set_density

# Further checks of the reverse-question signals (follow-up experiment):
# 1. Gold passages only as negatives: a gold passage answers the user's
#    question too, so it is the hardest benign case for question_focus.
# 2. Number of generated questions per passage (1 to 5).
# 3. Copied question: the Kind 1 poison is the Kind 2 poison with the
#    question in front, so the difference between the two shows how much
#    of question_focus comes from the generator copying the question.
# 4. Fewer poisoned passages per question (as evaluate_poison_count.py),
#    alone and combined with set density.
# 5. Detection rate and flagged gold passages at the 5% FPR threshold.
# 6. The worst cases with their generated questions, for reading by hand.
TRUST_DIR = Path("data_processed/trust")
SCORES_PATH_TRUST = TRUST_DIR / "trust_scores.csv"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUERIES_PATH = Path("data_processed/experiment_queries.json")
OUTPUT_PATH = TRUST_DIR / "reverse_question_analysis.json"
CASES_PATH = TRUST_DIR / "reverse_question_cases.csv"

SIGNALS = ("question_focus", "question_sameness")
CASES_PER_GROUP = 10
TEXT_PREVIEW_CHARS = 200


def main():
    with SCORES_PATH_TRUST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with SCORES_PATH.open(encoding="utf-8") as f:
        reverse = {r["passage_id"]: r for r in csv.DictReader(f)}

    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        generated = {
            r["passage_id"]: [r[f"question_{i + 1}"]
                              for i in range(N_QUESTIONS)]
            for r in csv.DictReader(f)
        }

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = {
            p["passage_id"]: p for p in (json.loads(line) for line in f)
        }

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    attacks = np.array([r["attack"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    gold = np.array([r["is_gold"] == "1" for r in rows])
    benign = attacks == "none"
    values = {
        name: np.array([float(reverse[r["passage_id"]][name]) for r in rows])
        for name in SIGNALS
    }
    retrieval = np.array([float(r["score_original"]) for r in rows])
    results = {}

    # 1. All benign passages or gold passages only as negatives.
    results["negatives"] = {}

    for attack in ATTACKS:
        results["negatives"][attack] = {}

        for negatives, mask in (("all_benign", benign), ("gold_only", gold)):
            keep = mask | (attacks == attack)
            density = np.array([float(r[f"set_density_{attack}"])
                                for r in rows])
            results["negatives"][attack][negatives] = {
                name: auroc(labels[keep], s[keep])
                for name, s in (*values.items(),
                                ("retrieval_score", retrieval),
                                ("set_density", density))
            }

    # 2. Number of generated questions per passage.
    question_emb, question_index = load_embeddings(EMBEDDING_NAME)
    query_emb, query_index = load_embeddings("query")
    results["questions_per_passage"] = {}

    for k in range(1, N_QUESTIONS + 1):
        pairs = np.array([
            question_signals(
                np.stack([question_emb[question_index[f"{r['passage_id']}#{i}"]]
                          for i in range(k)]),
                query_emb[query_index[f"{r['query_id']}_original"]],
            )
            for r in rows
        ])

        if k == N_QUESTIONS:
            assert np.allclose(pairs[:, 0], values["question_focus"])

        results["questions_per_passage"][str(k)] = {}

        for attack in ATTACKS:
            keep = benign | (attacks == attack)
            results["questions_per_passage"][str(k)][attack] = {
                "question_focus": auroc(labels[keep], pairs[keep, 0]),
                "question_sameness": (
                    auroc(labels[keep], pairs[keep, 1]) if k > 1 else None
                ),
            }

    # 3. Copied question: the same poisoned passages with and without the
    # question in front.
    def poisoned_values(attack, name):
        return {
            r["query_id"] + r["passage_id"].split("::")[1].split("_")[0]:
            values[name][i]
            for i, r in enumerate(rows) if r["attack"] == attack
        }

    results["copied_question"] = {}

    for name in SIGNALS:
        with_prefix = poisoned_values("prefix", name)
        without = poisoned_values("noprefix", name)
        assert with_prefix.keys() == without.keys()
        difference = np.array([with_prefix[k] - without[k]
                               for k in with_prefix])
        results["copied_question"][name] = {
            "benign_mean": float(values[name][benign].mean()),
            "with_prefix_mean": float(np.mean(list(with_prefix.values()))),
            "without_prefix_mean": float(np.mean(list(without.values()))),
            "share_higher_with_prefix": float((difference > 0).mean()),
        }

    # Share of passages with a generated question that is exactly the
    # user's question (lower case, without the question mark).
    def copies(pid, qid):
        user = queries[f"{qid}_original"].lower().strip(" ?")
        return any(q.lower().strip(" ?") == user for q in generated[pid])

    results["copied_question"]["exact_copy_share"] = {
        kind: float(np.mean([
            copies(r["passage_id"], r["query_id"])
            for r in rows if r["attack"] == kind
        ]))
        for kind in ("none", *ATTACKS)
    }

    # 4. Fewer poisoned passages per question.
    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")
    embeddings = np.stack([
        adv_emb[adv_index[r["passage_id"]]] if r["label"] == "1"
        else pool_emb[pool_index[passages[r["passage_id"]]["doc_id"]]]
        for r in rows
    ])
    rng = np.random.default_rng(SEED)
    results["poison_count"] = {}

    for attack in ATTACKS:
        results["poison_count"][attack] = {}

        for count in POISON_COUNTS:
            draws = {}

            for _ in range(N_DRAWS):
                keep = benign.copy()

                for qid in np.unique(groups):
                    poisoned = np.flatnonzero(
                        (groups == qid) & (attacks == attack)
                    )
                    keep[rng.choice(poisoned, size=count,
                                    replace=False)] = True

                kept = np.flatnonzero(keep)
                density = np.zeros(len(kept))

                for qid in np.unique(groups[kept]):
                    members = np.flatnonzero(groups[kept] == qid)
                    density[members] = set_density(
                        embeddings[kept[members]], SET_NEIGHBOURS
                    )

                y, g = labels[kept], groups[kept]
                focus = values["question_focus"][kept]
                scores = {
                    "question_focus": focus,
                    "set_density": density,
                    "set_density+focus": cross_validated_scores(
                        np.stack([density, focus], axis=1), y, g
                    ),
                    "retrieval+focus": cross_validated_scores(
                        np.stack([retrieval[kept], focus], axis=1), y, g
                    ),
                }

                for name, s in scores.items():
                    draws.setdefault(name, []).append(auroc(y, s))

            results["poison_count"][attack][str(count)] = {
                name: float(np.mean(v)) for name, v in draws.items()
            }

    # 5. Threshold that flags 5% of benign passages.
    results["threshold"] = {}

    for name in SIGNALS:
        threshold = float(np.quantile(values[name][benign], 1 - TARGET_FPR))
        flagged = values[name] > threshold
        results["threshold"][name] = {
            "threshold": threshold,
            "gold_flagged": int((flagged & gold).sum()),
            "gold_total": int(gold.sum()),
            **{f"tpr_{attack}": float(flagged[attacks == attack].mean())
               for attack in ATTACKS},
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 6. Worst cases: poisoned passages with the lowest focus and benign
    # passages with the highest focus.
    focus = values["question_focus"]
    cases = []

    for error, kinds, order in (
        ("missed", ATTACKS, 1), ("false_alarm", ("none",), -1),
    ):
        for kind in kinds:
            index = np.flatnonzero(attacks == kind)
            index = index[np.argsort(order * focus[index])][:CASES_PER_GROUP]

            for i in index:
                r = rows[i]
                cases.append({
                    "error": error,
                    "attack": kind,
                    "passage_id": r["passage_id"],
                    "is_gold": r["is_gold"],
                    "question_focus": round(focus[i], 3),
                    "question_sameness": round(
                        values["question_sameness"][i], 3
                    ),
                    "user_question": queries[f"{r['query_id']}_original"],
                    **{f"generated_{n + 1}": q for n, q in
                       enumerate(generated[r["passage_id"]])},
                    "passage_text":
                        passages[r["passage_id"]]["text"][:TEXT_PREVIEW_CHARS],
                })

    with CASES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)

    print("\nREVERSE QUESTIONS: FURTHER CHECKS")
    print("=" * 70)
    print("\n1. AUROC with all benign / gold-only negatives")

    for attack in ATTACKS:
        a, g = (results["negatives"][attack][n]
                for n in ("all_benign", "gold_only"))
        print(f"  {attack}")
        for name in a:
            print(f"    {name:<20}{a[name]:>8.3f}{g[name]:>8.3f}")

    print("\n2. Questions per passage (focus / sameness, Kind 1 and Kind 2)")

    for k, r in results["questions_per_passage"].items():
        line = "".join(
            f"{r[a]['question_focus']:>8.3f}"
            + (f"{r[a]['question_sameness']:>8.3f}"
               if r[a]["question_sameness"] is not None else f"{'-':>8}")
            for a in ATTACKS
        )
        print(f"  k={k}{line}")

    print("\n3. Copied question (mean benign / with prefix / without)")

    for name in SIGNALS:
        c = results["copied_question"][name]
        print(f"  {name:<20}{c['benign_mean']:.3f} / "
              f"{c['with_prefix_mean']:.3f} / {c['without_prefix_mean']:.3f}"
              f"  (higher with prefix: {c['share_higher_with_prefix']:.0%})")

    print("  exact copy of the user's question:",
          {k: f"{v:.0%}" for k, v in
           results["copied_question"]["exact_copy_share"].items()})

    print("\n4. Poisoned passages per question (AUROC)")

    for attack in ATTACKS:
        print(f"  {attack}")
        for count, r in results["poison_count"][attack].items():
            print(f"    {count}: " + "  ".join(
                f"{n} {v:.3f}" for n, v in r.items()
            ))

    print("\n5. At 5% false alarms")

    for name, t in results["threshold"].items():
        print(f"  {name}: Kind 1 {t['tpr_prefix']:.3f}, Kind 2 "
              f"{t['tpr_noprefix']:.3f}, gold flagged {t['gold_flagged']}"
              f" of {t['gold_total']}")

    print("\nCases written:", len(cases), "->", CASES_PATH)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
