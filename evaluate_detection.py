
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.trust import consistency_drop

TRUST_DIR = Path("data_processed/trust")
INPUT_PATH = TRUST_DIR / "trust_scores.csv"
OUTPUT_PATH = TRUST_DIR / "detection_results.json"

PARAPHRASES = ("para1", "para2", "para3")
ATTACKS = ("prefix", "noprefix")
# Which benign passages the poisoned passages are compared against.
NEGATIVES = ("all_benign", "gold_only")

# Combined scores: logistic regression over several signals, trained and
# tested on different questions (cross-validation grouped by question).
COMBINATIONS = {
    "combined_consistency": [
        "z_drop_k3", "support_drop_k3", "answer_disagree_k3",
    ],
    "combined_with_retrieval": [
        "z_drop_k3", "support_drop_k3", "answer_disagree_k3", "retrieval_z",
    ],
}

# Paired comparisons of AUROC (first method minus second method).
COMPARISONS = [
    ("z_drop_k3", "retrieval_z"),
    ("combined_consistency", "retrieval_z"),
    ("combined_with_retrieval", "retrieval_z"),
    ("combined_consistency", "z_drop_k3"),
]

TARGET_FPR = 0.05
N_FOLDS = 5
N_BOOTSTRAPS = 1000
SEED = 42


def auroc(labels, scores):
    ranks = rankdata(scores)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
        / (n_pos * n_neg)
    )


def tpr_at_fpr(labels, scores, fpr=TARGET_FPR):
    # Threshold that wrongly flags at most `fpr` of the benign passages.
    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    return float((scores[labels == 1] > threshold).mean())


def suspicion_scores(data, k):
    """All consistency signals, using the first k paraphrases."""
    names = PARAPHRASES[:k]

    def paraphrase_columns(prefix):
        return np.stack([data[f"{prefix}_{n}"] for n in names], axis=1)

    return {
        f"raw_drop_k{k}": consistency_drop(
            data["score_original"], paraphrase_columns("score")
        ),
        f"z_drop_k{k}": consistency_drop(
            data["z_original"], paraphrase_columns("z")
        ),
        f"rank_drop_k{k}": -consistency_drop(
            np.log(data["rank_original"]),
            np.log(paraphrase_columns("rank")),
        ),
        f"support_drop_k{k}": consistency_drop(
            data["support_original"], paraphrase_columns("support")
        ),
        f"answer_disagree_k{k}": (
            1 - paraphrase_columns("answer_f1").mean(axis=1)
        ),
    }


def cross_validated_scores(features, labels, groups):
    """Out-of-fold scores: each question is scored by a model that never
    saw that question during training."""
    scores = np.zeros(len(labels))

    for train, test in GroupKFold(n_splits=N_FOLDS).split(
        features, labels, groups
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(features[train], labels[train])
        scores[test] = model.decision_function(features[test])

    return scores


def main():
    with INPUT_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    text_columns = ("passage_id", "query_id", "attack")
    data = {
        key: np.array([
            row[key] if key in text_columns else float(row[key])
            for row in rows
        ])
        for key in rows[0]
    }

    labels = data["label"].astype(int)

    methods = {
        # Baselines that use the original question only (no paraphrases).
        "retrieval_score": data["score_original"],
        "retrieval_z": data["z_original"],
        "reader_support": data["support_original"],
        "perplexity": data["log_perplexity"],
        # Shorter passages are treated as more suspicious.
        "shortness": -data["num_words"],
    }

    for k in range(1, len(PARAPHRASES) + 1):
        methods.update(suspicion_scores(data, k))

    query_ids = sorted(set(data["query_id"]))
    rows_of_query = {
        qid: np.flatnonzero(data["query_id"] == qid) for qid in query_ids
    }

    rng = np.random.default_rng(SEED)
    bootstrap_samples = [
        np.concatenate([
            rows_of_query[qid]
            for qid in rng.choice(query_ids, size=len(query_ids))
        ])
        for _ in range(N_BOOTSTRAPS)
    ]

    results = {}

    for attack in ATTACKS:
        in_attack = (data["attack"] == "none") | (data["attack"] == attack)
        attack_methods = dict(methods)
        # Set-level baseline: depends on which passages are in the set.
        attack_methods["set_density"] = data[f"set_density_{attack}"]

        # Perplexity with the effect of length removed: the part of
        # perplexity that a straight line on log length cannot explain.
        # The line is fitted without using the labels.
        log_length = np.log(data["num_words"])
        slope, intercept = np.polyfit(
            log_length[in_attack], data["log_perplexity"][in_attack], 1
        )
        attack_methods["perplexity_length_adjusted"] = (
            data["log_perplexity"] - (slope * log_length + intercept)
        )

        for name, feature_names in COMBINATIONS.items():
            features = np.stack(
                [methods[f][in_attack] for f in feature_names], axis=1
            )
            combined = np.full(len(labels), np.nan)
            combined[in_attack] = cross_validated_scores(
                features, labels[in_attack], data["query_id"][in_attack]
            )
            attack_methods[name] = combined

        for negatives in NEGATIVES:
            benign = data["attack"] == "none"

            if negatives == "gold_only":
                benign = benign & (data["is_gold"] == 1)

            keep = benign | (data["attack"] == attack)

            setting = f"{attack}/{negatives}"
            results[setting] = {
                "poisoned": int((keep & (labels == 1)).sum()),
                "benign": int((keep & (labels == 0)).sum()),
                "methods": {},
            }

            kept_samples = [
                sample[keep[sample]] for sample in bootstrap_samples
            ]
            boot_aurocs = {}

            for name, scores in attack_methods.items():
                y, s = labels[keep], scores[keep]

                # Bootstrap over questions for 95% confidence intervals.
                boot = np.array([
                    auroc(labels[sample], scores[sample])
                    for sample in kept_samples
                ])
                boot_aurocs[name] = boot

                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

                results[setting]["methods"][name] = {
                    "auroc": auroc(y, s),
                    "auroc_ci_95": [float(ci_low), float(ci_high)],
                    "auprc": float(average_precision_score(y, s)),
                    "tpr_at_5pct_fpr": tpr_at_fpr(y, s),
                }

            # Paired bootstrap test: both methods are scored on the same
            # resampled questions, so the difference is compared fairly.
            results[setting]["comparisons"] = {}

            for first, second in COMPARISONS:
                differences = boot_aurocs[first] - boot_aurocs[second]
                ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
                p_value = 2 * min(
                    (differences <= 0).mean(), (differences >= 0).mean()
                )
                observed = (
                    results[setting]["methods"][first]["auroc"]
                    - results[setting]["methods"][second]["auroc"]
                )

                results[setting]["comparisons"][f"{first} - {second}"] = {
                    "auroc_difference": observed,
                    "ci_95": [float(ci_low), float(ci_high)],
                    "p_value": float(min(1.0, p_value)),
                }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nPOISONED PASSAGE DETECTION")
    print("=" * 82)

    for setting, result in results.items():
        print(
            f"\nSetting: {setting} "
            f"({result['poisoned']} poisoned, {result['benign']} benign)"
        )
        print(f"{'method':<26}{'AUROC':>8}{'95% CI':>20}"
              f"{'AUPRC':>9}{'TPR@5%FPR':>12}")

        for name, m in result["methods"].items():
            # The k=1 and k=2 rows are shown in the ablation table below.
            if name.endswith(("_k1", "_k2")):
                continue

            ci = f"[{m['auroc_ci_95'][0]:.3f}, {m['auroc_ci_95'][1]:.3f}]"
            print(f"{name:<26}{m['auroc']:>8.3f}{ci:>20}"
                  f"{m['auprc']:>9.3f}{m['tpr_at_5pct_fpr']:>12.3f}")

    print("\nPAIRED COMPARISONS (AUROC difference, bootstrap over questions)")
    print("=" * 82)

    for setting, result in results.items():
        print(f"\nSetting: {setting}")

        for name, c in result["comparisons"].items():
            ci = f"[{c['ci_95'][0]:+.3f}, {c['ci_95'][1]:+.3f}]"
            print(f"{name:<46}{c['auroc_difference']:>+8.3f}{ci:>20}"
                  f"   p={c['p_value']:.3f}")

    print("\nABLATION: AUROC BY NUMBER OF PARAPHRASES (all benign)")
    print("=" * 82)
    print(f"{'signal':<20}{'attack':<12}{'k=1':>8}{'k=2':>8}{'k=3':>8}")

    for signal in ("raw_drop", "z_drop", "rank_drop",
                   "support_drop", "answer_disagree"):
        for attack in ATTACKS:
            values = [
                results[f"{attack}/all_benign"]["methods"]
                [f"{signal}_k{k}"]["auroc"]
                for k in (1, 2, 3)
            ]
            print(f"{signal:<20}{attack:<12}"
                  + "".join(f"{v:>8.3f}" for v in values))

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
