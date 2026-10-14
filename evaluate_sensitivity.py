
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import (
    SET_NEIGHBOURS, VARIANTS, load_embeddings, reference_masks,
)
from evaluate_detection import (
    ATTACKS, COMBINATIONS, TARGET_FPR, auroc, cross_validated_scores,
    tpr_at_fpr,
)
from src.trust import REFERENCE_TOP_N, set_density

# Sensitivity of the results to the fixed design choices (run after the
# main results; the values used in the main results were not changed):
# 0. Which passages the z-score reference comes from (see
#    compute_trust_scores.reference_masks).
# 1. REFERENCE_TOP_N: how many top clean passages define the z-score.
# 2. How the paraphrase scores are combined in z_drop (mean is used).
# 3. SET_NEIGHBOURS: neighbours used by the set-density baseline.
# 4. The false-positive rate used for the detection threshold.
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
OUTPUT_PATH = TRUST_DIR / "sensitivity_results.json"

TOP_N_VALUES = (10, 25, 50, 100, 200, 500)
AGGREGATIONS = {
    "mean": lambda z: z.mean(axis=1),
    "median": lambda z: np.median(z, axis=1),
    "lowest": lambda z: z.min(axis=1),
    "highest": lambda z: z.max(axis=1),
}
NEIGHBOUR_VALUES = (1, 2, 3, 4, 5)
FPR_VALUES = (0.01, 0.05, 0.10)


def main():
    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    doc_ids = {p["passage_id"]: p["doc_id"] for p in passages}

    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")
    query_emb, query_index = load_embeddings("query")
    references = reference_masks(pool_index, passages)

    attacks = np.array([r["attack"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    scores = np.array([[float(r[f"score_{v}"]) for v in VARIANTS]
                       for r in rows])
    z = np.array([[float(r[f"z_{v}"]) for v in VARIANTS] for r in rows])
    embeddings = np.stack([
        adv_emb[adv_index[r["passage_id"]]] if r["label"] == "1"
        else pool_emb[pool_index[doc_ids[r["passage_id"]]]]
        for r in rows
    ])

    def by_attack(values):
        """AUROC on Kind 1 and Kind 2 (all benign passages as negatives)."""
        result = {}

        for attack in ATTACKS:
            keep = (attacks == "none") | (attacks == attack)
            column = values[attack] if isinstance(values, dict) else values
            result[attack] = auroc(labels[keep], column[keep])

        return result

    results = {"chosen": {"reference_top_n": REFERENCE_TOP_N,
                          "aggregation": "mean",
                          "set_neighbours": SET_NEIGHBOURS,
                          "target_fpr": TARGET_FPR}}

    # 1. z-score reference size. The sorted reference scores of every
    # query wording are computed once per question.
    top_scores = {}

    for qid in np.unique(groups):
        variants = np.stack([
            query_emb[query_index[f"{qid}_{v}"]] for v in VARIANTS
        ])
        top_scores[qid] = -np.sort(
            -(variants @ pool_emb[references[qid]].T), axis=1
        )

    # 0. Reference choice: the whole pool (the first version, unfair to
    # the paraphrases), the pool without the question's own passages
    # (used), and no reference at all (raw scores).
    whole_pool = {}

    for qid in np.unique(groups):
        variants = np.stack([
            query_emb[query_index[f"{qid}_{v}"]] for v in VARIANTS
        ])
        top = -np.sort(-(variants @ pool_emb.T), axis=1)[:, :REFERENCE_TOP_N]
        whole_pool[qid] = (top.mean(axis=1), top.std(axis=1))

    z_whole = (scores - np.stack([whole_pool[q][0] for q in groups])) \
        / np.stack([whole_pool[q][1] for q in groups])
    results["reference"] = {
        "whole_pool": {
            "retrieval": by_attack(z_whole[:, 0]),
            "drop": by_attack(z_whole[:, 0] - z_whole[:, 1:].mean(axis=1)),
        },
        "pool_without_own_passages": {
            "retrieval": by_attack(z[:, 0]),
            "drop": by_attack(z[:, 0] - z[:, 1:].mean(axis=1)),
        },
        "none_raw_scores": {
            "retrieval": by_attack(scores[:, 0]),
            "drop": by_attack(scores[:, 0] - scores[:, 1:].mean(axis=1)),
        },
    }

    results["reference_top_n"] = {}

    for top_n in TOP_N_VALUES:
        mean = np.stack([top_scores[q][:, :top_n].mean(axis=1)
                         for q in groups])
        std = np.stack([top_scores[q][:, :top_n].std(axis=1)
                        for q in groups])
        z_n = (scores - mean) / std
        assert top_n != REFERENCE_TOP_N or np.allclose(z_n, z, atol=1e-4)
        results["reference_top_n"][str(top_n)] = by_attack(
            z_n[:, 0] - z_n[:, 1:].mean(axis=1)
        )

    # 2. Combining the three paraphrase scores.
    results["aggregation"] = {
        name: by_attack(z[:, 0] - combine(z[:, 1:]))
        for name, combine in AGGREGATIONS.items()
    }

    # 3. Set-density neighbours, inside each question's candidate set.
    results["set_neighbours"] = {}

    for neighbours in NEIGHBOUR_VALUES:
        density = {}

        for attack in ATTACKS:
            density[attack] = np.full(len(rows), np.nan)

            for qid in np.unique(groups):
                members = np.flatnonzero(
                    (groups == qid)
                    & ((attacks == "none") | (attacks == attack))
                )
                density[attack][members] = set_density(
                    embeddings[members], neighbours
                )

            if neighbours == SET_NEIGHBOURS:
                saved = np.array([float(r[f"set_density_{attack}"])
                                  for r in rows])
                keep = ~np.isnan(saved)
                assert np.allclose(density[attack][keep], saved[keep],
                                   atol=1e-5)

        results["set_neighbours"][str(neighbours)] = by_attack(density)

    # 4. Detection rate at different false-positive rates.
    methods = {
        "retrieval_score": scores[:, 0],
        "set_density": None,
        "shortness": -np.array([float(r["num_words"]) for r in rows]),
        "perplexity": np.array([float(r["log_perplexity"]) for r in rows]),
        "reader_support": np.array([float(r["support_original"])
                                    for r in rows]),
        "z_drop_k3": np.array([float(r["z_drop"]) for r in rows]),
    }
    columns = {
        "z_drop_k3": "z_drop", "support_drop_k3": "support_drop",
        "answer_disagree_k3": "answer_disagreement",
        "retrieval_score": "score_original",
    }
    results["tpr_at_fpr"] = {}

    for attack in ATTACKS:
        keep = (attacks == "none") | (attacks == attack)
        attack_methods = dict(methods)
        attack_methods["set_density"] = np.array(
            [float(r[f"set_density_{attack}"]) for r in rows]
        )

        for name, features in COMBINATIONS.items():
            x = np.stack([
                np.array([float(r[columns[f]]) for r in rows])[keep]
                for f in features
            ], axis=1)
            combined = np.full(len(rows), np.nan)
            combined[keep] = cross_validated_scores(
                x, labels[keep], groups[keep]
            )
            attack_methods[name] = combined

        results["tpr_at_fpr"][attack] = {
            name: {str(fpr): tpr_at_fpr(labels[keep], s[keep], fpr)
                   for fpr in FPR_VALUES}
            for name, s in attack_methods.items()
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSENSITIVITY TO DESIGN CHOICES (AUROC, Kind 1 / Kind 2)")
    print("=" * 60)
    print("\nz-score reference      retrieval          drop")

    for name, r in results["reference"].items():
        print(f"  {name:<26}"
              f"{r['retrieval']['prefix']:>7.3f}{r['retrieval']['noprefix']:>7.3f}"
              f"{r['drop']['prefix']:>9.3f}{r['drop']['noprefix']:>7.3f}")

    for title, key, chosen in (
        ("z-score reference size (top N)", "reference_top_n",
         str(REFERENCE_TOP_N)),
        ("Combining the paraphrase scores", "aggregation", "mean"),
        ("Set-density neighbours", "set_neighbours", str(SET_NEIGHBOURS)),
    ):
        print(f"\n{title}")

        for value, r in results[key].items():
            mark = "  (used)" if value == chosen else ""
            print(f"  {value:<10}{r['prefix']:>8.3f}{r['noprefix']:>8.3f}"
                  f"{mark}")

    print("\nDetection rate (TPR) at different false-positive rates")

    for attack in ATTACKS:
        print(f"\n  {attack:<26}"
              + "".join(f"{f'FPR {fpr:.0%}':>10}" for fpr in FPR_VALUES))

        for name, values in results["tpr_at_fpr"][attack].items():
            print(f"  {name:<26}"
                  + "".join(f"{values[str(fpr)]:>10.3f}"
                            for fpr in FPR_VALUES))

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
