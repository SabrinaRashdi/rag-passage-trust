
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from compute_trust_scores import VARIANTS
from evaluate_detection import ATTACKS, N_BOOTSTRAPS, SEED, auroc

# Why relevance consistency (z_drop) does not separate poisoned from
# benign passages. Four checks:
# 1. Words: how many of the question's content words each passage
#    contains, for the original question and for the paraphrases.
# 2. Selection: benign passages are in the set because they ranked high
#    for the original wording, so they are expected to drop anyway.
# 3. Score level: passages with a higher original z-score drop more. Is
#    z_drop still useful when passages start from the same z-score?
# 4. Effect size of the difference in z_drop.
# This analysis was done after the main results.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

SCORES_PATH = TRUST_DIR / "trust_scores.csv"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
OUTPUT_PATH = TRUST_DIR / "why_consistency_fails.json"

# Passages are compared inside bins of equal size on the original z-score.
N_LEVEL_BINS = 10
# Bins with fewer benign or poisoned passages than this are not used.
MIN_PER_GROUP = 10
TYPES = ("none", *ATTACKS)


def content_words(text):
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in ENGLISH_STOP_WORDS
    }


def correlation(x, y):
    result = spearmanr(x, y)
    return {"spearman": float(result[0]), "p_value": float(result[1])}


def main():
    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with EVAL_PATH.open(encoding="utf-8") as f:
        texts = {
            p["passage_id"]: p["text"]
            for p in (json.loads(line) for line in f)
        }

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    attacks = np.array([r["attack"] for r in rows])
    query_ids = np.array([r["query_id"] for r in rows])
    top10 = np.array([r["retrieved_top10"] == "1" for r in rows])
    gold = np.array([r["is_gold"] == "1" for r in rows])
    scores = np.array([[float(r[f"score_{v}"]) for v in VARIANTS]
                       for r in rows])
    z = np.array([[float(r[f"z_{v}"]) for v in VARIANTS] for r in rows])
    z_drop = np.array([float(r["z_drop"]) for r in rows])
    z_original = z[:, 0]
    benign = attacks == "none"
    questions = sorted(set(query_ids))

    rng = np.random.default_rng(SEED)
    rows_of_question = {q: np.flatnonzero(query_ids == q) for q in questions}
    bootstrap = [
        np.concatenate([rows_of_question[q]
                        for q in rng.choice(questions, len(questions))])
        for _ in range(N_BOOTSTRAPS)
    ]

    def with_ci(statistic, keep):
        """Statistic on the passages in keep, with a 95% CI from
        resampling questions."""
        values = [statistic(sample[keep[sample]]) for sample in bootstrap]
        low, high = np.percentile(values, [2.5, 97.5])
        return {"value": statistic(np.flatnonzero(keep)),
                "ci_95": [float(low), float(high)]}

    # 1. Content-word overlap between each passage and each wording.
    question_words = {
        (q, v): content_words(queries[f"{q}_{v}"])
        for q in questions for v in VARIANTS
    }
    overlap = np.array([
        [len(question_words[r["query_id"], v]
             & content_words(texts[r["passage_id"]]))
         / len(question_words[r["query_id"], v]) for v in VARIANTS]
        for r in rows
    ])
    # Share of the original question's content words that each
    # paraphrase keeps.
    retention = {
        q: float(np.mean([
            len(question_words[q, "original"] & question_words[q, v])
            / len(question_words[q, "original"]) for v in VARIANTS[1:]
        ]))
        for q in questions
    }
    retention_of_row = np.array([retention[q] for q in query_ids])
    median_retention = float(np.median(list(retention.values())))

    results = {
        "paraphrase_word_retention": {
            "mean": float(np.mean(list(retention.values()))),
            "median": median_retention,
            "questions_all_words_kept": int(sum(
                v == 1 for v in retention.values()
            )),
        },
        "by_type": {},
    }

    for kind in TYPES:
        m = attacks == kind
        results["by_type"][kind] = {
            "passages": int(m.sum()),
            "word_overlap_original": float(overlap[m, 0].mean()),
            "word_overlap_paraphrases": float(overlap[m, 1:].mean()),
            "raw_score_original": float(scores[m, 0].mean()),
            "raw_score_paraphrases": float(scores[m, 1:].mean()),
            "share_of_raw_score_kept": float(
                (scores[m, 1:].mean(axis=1) / scores[m, 0]).mean()
            ),
            "z_original": float(z_original[m].mean()),
            "z_paraphrases": float(z[m, 1:].mean()),
            "z_drop_mean": float(z_drop[m].mean()),
            "z_drop_std": float(z_drop[m].std()),
            "z_drop_relative_to_z_original": float(
                (z_drop[m] / z_original[m]).mean()
            ),
            "z_drop_vs_z_original": correlation(z_original[m], z_drop[m]),
        }

    # 2. Selection effect inside the benign passages.
    results["benign_selection"] = {
        "z_drop_retrieved_top10": float(z_drop[benign & top10].mean()),
        "z_drop_not_in_top10": float(z_drop[benign & ~top10].mean()),
        "passages_top10": int((benign & top10).sum()),
        "passages_not_in_top10": int((benign & ~top10).sum()),
        "z_drop_gold": float(z_drop[benign & gold].mean()),
    }

    # 3. and 4., for each attack version.
    for attack in ATTACKS:
        keep = benign | (attacks == attack)
        labels = (attacks == attack).astype(int)

        # Bins on the original z-score, from all passages in the set (no
        # labels used).
        edges = np.quantile(
            z_original[keep], np.linspace(0, 1, N_LEVEL_BINS + 1)
        )
        level_bin = np.clip(
            np.digitize(z_original, edges[1:-1]), 0, N_LEVEL_BINS - 1
        )
        # A straight line of z_drop on the original z-score, also fitted
        # without labels; the residual is the drop not explained by the
        # starting level.
        slope, intercept = np.polyfit(z_original[keep], z_drop[keep], 1)
        residual = z_drop - (slope * z_original + intercept)

        def within_level_auroc(index):
            values = []

            for b in range(N_LEVEL_BINS):
                in_bin = index[level_bin[index] == b]
                n_poisoned = labels[in_bin].sum()
                if min(n_poisoned, len(in_bin) - n_poisoned) >= MIN_PER_GROUP:
                    values.append(auroc(labels[in_bin], z_drop[in_bin]))

            return float(np.mean(values))

        bins = []

        for b in range(N_LEVEL_BINS):
            in_bin = keep & (level_bin == b)
            bins.append({
                "z_original_range": [float(edges[b]), float(edges[b + 1])],
                "benign": int((in_bin & benign).sum()),
                "poisoned": int((in_bin & (labels == 1)).sum()),
                "used": bool(min((in_bin & benign).sum(),
                                 (in_bin & (labels == 1)).sum())
                             >= MIN_PER_GROUP),
                "z_drop_benign": float(z_drop[in_bin & benign].mean()),
                "z_drop_poisoned": float(z_drop[in_bin & (labels == 1)].mean()),
            })

        low = retention_of_row < median_retention
        poisoned = labels == 1
        results[attack] = {
            "auroc_z_drop": with_ci(
                lambda i: auroc(labels[i], z_drop[i]), keep
            ),
            "auroc_z_original": with_ci(
                lambda i: auroc(labels[i], z_original[i]), keep
            ),
            "auroc_within_level_bins": with_ci(within_level_auroc, keep),
            "auroc_residual_after_level": with_ci(
                lambda i: auroc(labels[i], residual[i]), keep
            ),
            "residual_line": {"slope": float(slope),
                              "intercept": float(intercept)},
            "level_bins": bins,
            "level_bins_used": int(sum(b["used"] for b in bins)),
            "cohens_d_z_drop": float(
                (z_drop[poisoned].mean() - z_drop[benign].mean())
                / np.sqrt((z_drop[poisoned].var() + z_drop[benign].var())
                          / 2)
            ),
            "auroc_low_word_retention": with_ci(
                lambda i: auroc(labels[i], z_drop[i]), keep & low
            ),
            "auroc_high_word_retention": with_ci(
                lambda i: auroc(labels[i], z_drop[i]), keep & ~low
            ),
            "z_drop_vs_word_overlap_drop": correlation(
                (overlap[:, 0] - overlap[:, 1:].mean(axis=1))[poisoned],
                z_drop[poisoned],
            ),
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nWHY RELEVANCE CONSISTENCY FAILS")
    print("=" * 70)
    r = results["paraphrase_word_retention"]
    print(f"Paraphrases keep {r['mean']:.0%} of the question's content "
          f"words (all of them in {r['questions_all_words_kept']} "
          f"questions)")

    print(f"\n{'':<34}" + "".join(f"{k:>12}" for k in TYPES))

    for key in ("word_overlap_original", "word_overlap_paraphrases",
                "share_of_raw_score_kept", "z_original", "z_paraphrases",
                "z_drop_mean", "z_drop_std",
                "z_drop_relative_to_z_original"):
        print(f"{key:<34}" + "".join(
            f"{results['by_type'][k][key]:>12.3f}" for k in TYPES
        ))

    s = results["benign_selection"]
    print(f"\nBenign z_drop: in top 10 {s['z_drop_retrieved_top10']:.2f} "
          f"(n={s['passages_top10']}), not in top 10 "
          f"{s['z_drop_not_in_top10']:.2f} (n={s['passages_not_in_top10']}),"
          f" gold {s['z_drop_gold']:.2f}")

    for attack in ATTACKS:
        a = results[attack]
        print(f"\nAttack: {attack}   (Cohen's d of z_drop: "
              f"{a['cohens_d_z_drop']:+.2f})")

        for key in ("auroc_z_drop", "auroc_z_original",
                    "auroc_within_level_bins", "auroc_residual_after_level",
                    "auroc_low_word_retention", "auroc_high_word_retention"):
            v = a[key]
            print(f"  {key:<30}{v['value']:.3f}  "
                  f"[{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}]")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
