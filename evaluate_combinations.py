
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compute_trust_scores import SET_NEIGHBOURS, load_embeddings
from evaluate_detection import (
    ATTACKS, N_BOOTSTRAPS, N_FOLDS, SEED, auroc, cross_validated_scores,
    tpr_at_fpr,
)
from evaluate_poison_count import N_DRAWS, POISON_COUNTS
from src.trust import set_density

# Exploratory (planned after the earlier results): does our paraphrase
# signal add anything when it is combined with the baselines? The list of
# combinations was fixed before running this script, and all are reported.
# "ours" = the three consistency signals of combined_consistency.
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
ADAPTIVE_PATH = TRUST_DIR / "adaptive_attack_scores.csv"
OUTPUT_PATH = TRUST_DIR / "combination_results.json"

OURS = ["z_drop", "support_drop", "answer_disagreement"]
COMBINATIONS = {
    "ours": OURS,
    "ours+retrieval": OURS + ["z_original"],
    "ours+set_density": OURS + ["set_density"],
    "ours+reader": OURS + ["support_original"],
    "ours+all_three": OURS + ["z_original", "set_density",
                              "support_original"],
    "all_three_without_ours": ["z_original", "set_density",
                               "support_original"],
}
# Single baselines for reference (no training).
SINGLE = {
    "retrieval_z": "z_original",
    "set_density": "set_density",
    "reader_support": "support_original",
}
# Paired comparisons: first minus second.
COMPARISONS = [
    ("ours+all_three", "all_three_without_ours"),
    ("ours+set_density", "set_density"),
    ("ours+retrieval", "retrieval_z"),
]
KNOWN_ATTACK = "prefix"


def paired_difference(labels, groups, first, second, rng):
    questions = np.unique(groups)
    rows_of = {q: np.flatnonzero(groups == q) for q in questions}
    differences = []

    for _ in range(N_BOOTSTRAPS):
        sample = np.concatenate([
            rows_of[q] for q in rng.choice(questions, size=len(questions))
        ])
        differences.append(
            auroc(labels[sample], first[sample])
            - auroc(labels[sample], second[sample])
        )

    differences = np.array(differences)
    low, high = np.percentile(differences, [2.5, 97.5])

    return {
        "auroc_difference": auroc(labels, first) - auroc(labels, second),
        "ci_95": [float(low), float(high)],
        "p_value": float(min(1.0, 2 * min(
            (differences <= 0).mean(), (differences >= 0).mean()
        ))),
    }


def all_scores(features, labels, groups):
    """Scores of every method; combinations are cross-validated."""
    scores = {
        name: features[column] for name, column in SINGLE.items()
    }

    for name, columns in COMBINATIONS.items():
        scores[name] = cross_validated_scores(
            np.stack([features[c] for c in columns], axis=1), labels, groups
        )

    return scores


def evaluate(labels, groups, scores, rng):
    return {
        "methods": {
            name: {"auroc": auroc(labels, s),
                   "tpr_at_5pct_fpr": tpr_at_fpr(labels, s)}
            for name, s in scores.items()
        },
        "comparisons": {
            f"{a} - {b}": paired_difference(
                labels, groups, scores[a], scores[b], rng
            )
            for a, b in COMPARISONS
        },
    }


def main():
    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with EVAL_PATH.open(encoding="utf-8") as f:
        doc_ids = {
            p["passage_id"]: p.get("doc_id")
            for p in (json.loads(line) for line in f)
        }

    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")
    embeddings = np.stack([
        adv_emb[adv_index[r["passage_id"]]] if r["label"] == "1"
        else pool_emb[pool_index[doc_ids[r["passage_id"]]]]
        for r in rows
    ])

    attacks = np.array([r["attack"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    columns = {
        name: np.array([float(r[name]) for r in rows])
        for name in ("z_original", "support_original", *OURS)
    }

    rng = np.random.default_rng(SEED)
    results = {"main": {}, "poison_count": {}}

    # 1. Main setting: 5 poisoned passages per question.
    for attack in ATTACKS:
        keep = (attacks == "none") | (attacks == attack)
        features = {n: v[keep] for n, v in columns.items()}
        features["set_density"] = np.array(
            [float(r[f"set_density_{attack}"]) for r in rows]
        )[keep]
        scores = all_scores(features, labels[keep], groups[keep])
        results["main"][attack] = evaluate(
            labels[keep], groups[keep], scores, rng
        )

    # 2. Fewer poisoned passages per question (as evaluate_poison_count.py).
    for attack in ATTACKS:
        results["poison_count"][attack] = {}

        for count in POISON_COUNTS:
            draws = {}

            for _ in range(N_DRAWS):
                keep = attacks == "none"

                for qid in np.unique(groups):
                    poisoned = np.flatnonzero(
                        (groups == qid) & (attacks == attack)
                    )
                    keep[rng.choice(poisoned, size=count, replace=False)] = True

                kept = np.flatnonzero(keep)
                density = np.zeros(len(kept))

                for qid in np.unique(groups[kept]):
                    members = np.flatnonzero(groups[kept] == qid)
                    density[members] = set_density(
                        embeddings[kept[members]], SET_NEIGHBOURS
                    )

                features = {n: v[kept] for n, v in columns.items()}
                features["set_density"] = density
                scores = all_scores(features, labels[kept], groups[kept])

                for name, s in scores.items():
                    draws.setdefault(name, []).append(auroc(labels[kept], s))

            results["poison_count"][attack][str(count)] = {
                name: float(np.mean(v)) for name, v in draws.items()
            }

    # 3. Reworded attack: trained on the normal attack (Kind 1) only.
    with ADAPTIVE_PATH.open(encoding="utf-8") as f:
        adaptive = list(csv.DictReader(f))

    test_labels = np.array([int(r["label"]) for r in adaptive])
    test_groups = np.array([r["query_id"] for r in adaptive])
    test = {
        name: np.array([float(r[name]) for r in adaptive])
        for name in ("z_original", "support_original", "set_density", *OURS)
    }

    known = (attacks == "none") | (attacks == KNOWN_ATTACK)
    train = {n: v[known] for n, v in columns.items()}
    train["set_density"] = np.array(
        [float(r[f"set_density_{KNOWN_ATTACK}"]) for r in rows]
    )[known]
    train_labels, train_groups = labels[known], groups[known]

    scores = {name: test[column] for name, column in SINGLE.items()}

    for name, names in COMBINATIONS.items():
        x_train = np.stack([train[c] for c in names], axis=1)
        x_test = np.stack([test[c] for c in names], axis=1)
        combined = np.zeros(len(adaptive))

        # Each question is scored by a model that never saw it.
        for fit_rows, held_out in GroupKFold(n_splits=N_FOLDS).split(
            x_train, train_labels, train_groups
        ):
            model = make_pipeline(StandardScaler(), LogisticRegression())
            model.fit(x_train[fit_rows], train_labels[fit_rows])
            target = np.isin(test_groups, np.unique(train_groups[held_out]))
            combined[target] = model.decision_function(x_test[target])

        scores[name] = combined

    results["reworded_trained_on_kind1"] = evaluate(
        test_labels, test_groups, scores, rng
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    names = [*SINGLE, *COMBINATIONS]
    print("\nCOMBINING OUR SIGNAL WITH THE BASELINES (AUROC, exploratory)")
    print("=" * 86)
    print(f"{'method':<26}{'Kind 1':>9}{'Kind 2':>9}"
          f"{'K1 n=1':>9}{'K1 n=2':>9}{'K2 n=1':>9}{'K2 n=2':>9}"
          f"{'reworded':>10}")

    for name in names:
        print(
            f"{name:<26}"
            f"{results['main']['prefix']['methods'][name]['auroc']:>9.3f}"
            f"{results['main']['noprefix']['methods'][name]['auroc']:>9.3f}"
            f"{results['poison_count']['prefix']['1'][name]:>9.3f}"
            f"{results['poison_count']['prefix']['2'][name]:>9.3f}"
            f"{results['poison_count']['noprefix']['1'][name]:>9.3f}"
            f"{results['poison_count']['noprefix']['2'][name]:>9.3f}"
            f"{results['reworded_trained_on_kind1']['methods'][name]['auroc']:>10.3f}"
        )

    print("\nPAIRED COMPARISONS (AUROC difference, bootstrap over questions)")

    for setting, values in (
        ("Kind 1", results["main"]["prefix"]),
        ("Kind 2", results["main"]["noprefix"]),
        ("reworded", results["reworded_trained_on_kind1"]),
    ):
        for name, c in values["comparisons"].items():
            print(f"  {setting:<10}{name:<46}{c['auroc_difference']:>+8.3f}"
                  f"  [{c['ci_95'][0]:+.3f}, {c['ci_95'][1]:+.3f}]"
                  f"  p={c['p_value']:.3f}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
