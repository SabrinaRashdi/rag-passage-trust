
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from src.trust import consistency_drop

TRUST_DIR = Path("data_processed/trust")
INPUT_PATH = TRUST_DIR / "trust_scores.csv"
OUTPUT_PATH = TRUST_DIR / "detection_results.json"

PARAPHRASES = ("para1", "para2", "para3")
ATTACKS = ("prefix", "noprefix")
# Which benign passages the poisoned passages are compared against.
NEGATIVES = ("all_benign", "gold_only")

TARGET_FPR = 0.05
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
    """All detection scores, using the first k paraphrases."""
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
    }


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

    methods = {
        # Baseline without paraphrases: the plain retrieval score.
        "retrieval_score": data["score_original"],
        "retrieval_z": data["z_original"],
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
        for negatives in NEGATIVES:
            benign = data["attack"] == "none"

            if negatives == "gold_only":
                benign = benign & (data["is_gold"] == 1)

            keep = benign | (data["attack"] == attack)
            labels = data["label"].astype(int)

            setting = f"{attack}/{negatives}"
            results[setting] = {
                "poisoned": int((keep & (labels == 1)).sum()),
                "benign": int((keep & (labels == 0)).sum()),
                "methods": {},
            }

            for name, scores in methods.items():
                y, s = labels[keep], scores[keep]

                # Bootstrap over questions for 95% confidence intervals.
                boot = []

                for sample in bootstrap_samples:
                    sample = sample[keep[sample]]
                    boot.append(auroc(labels[sample], scores[sample]))

                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

                results[setting]["methods"][name] = {
                    "auroc": auroc(y, s),
                    "auroc_ci_95": [float(ci_low), float(ci_high)],
                    "auprc": float(average_precision_score(y, s)),
                    "tpr_at_5pct_fpr": tpr_at_fpr(y, s),
                }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nPOISONED PASSAGE DETECTION")
    print("=" * 78)

    for setting, result in results.items():
        print(
            f"\nSetting: {setting} "
            f"({result['poisoned']} poisoned, {result['benign']} benign)"
        )
        print(f"{'method':<18}{'AUROC':>8}{'95% CI':>20}"
              f"{'AUPRC':>9}{'TPR@5%FPR':>12}")

        for name, m in result["methods"].items():
            ci = f"[{m['auroc_ci_95'][0]:.3f}, {m['auroc_ci_95'][1]:.3f}]"
            print(f"{name:<18}{m['auroc']:>8.3f}{ci:>20}"
                  f"{m['auprc']:>9.3f}{m['tpr_at_5pct_fpr']:>12.3f}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
