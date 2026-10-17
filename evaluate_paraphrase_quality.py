
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from check_paraphrases_nli import OUTPUT_PATH as NLI_PATH
from check_paraphrases_nli import REVIEW_PATHS
from evaluate_detection import (
    ATTACKS, N_BOOTSTRAPS, PARAPHRASES, SEED, auroc, tpr_at_fpr,
)

# Manual paraphrase review: how much the two reviewers agree, how many
# paraphrases keep the meaning, and whether the relevance-consistency score
# changes when only the paraphrases judged to keep the meaning are used.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

SCORES_PATH = TRUST_DIR / "trust_scores.csv"
LABELS_PATH = DATA_DIR / "paraphrase_review_combined.csv"
OUTPUT_PATH = TRUST_DIR / "paraphrase_quality_results.json"


def load_review(path):
    with path.open(encoding="utf-8-sig") as f:
        return {
            row["item_id"]: row["same_meaning"].strip().lower() == "yes"
            for row in csv.DictReader(f)
        }


def agreement(first, second):
    first, second = np.array(first), np.array(second)
    return {
        "rows": len(first),
        "agreement": float((first == second).mean()),
        "cohen_kappa": float(cohen_kappa_score(first, second)),
    }


def drop(z_original, z_paraphrases, use):
    """z_drop using only the paraphrases where `use` is True
    (NaN for a passage whose question has no such paraphrase)."""
    count = use.sum(axis=1)
    total = np.where(use, z_paraphrases, 0).sum(axis=1)
    mean = np.divide(
        total, count, out=np.full(len(count), np.nan), where=count > 0
    )
    return z_original - mean


def compare(labels, groups, first, second, rng):
    """AUROC of two scores on the same passages, with a paired bootstrap
    over questions for the difference (first minus second)."""
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
        "questions": len(questions),
        "poisoned": int(labels.sum()),
        "benign": int((labels == 0).sum()),
        "auroc_first": auroc(labels, first),
        "auroc_second": auroc(labels, second),
        "tpr_at_5pct_fpr_first": tpr_at_fpr(labels, first),
        "tpr_at_5pct_fpr_second": tpr_at_fpr(labels, second),
        "auroc_difference": auroc(labels, first) - auroc(labels, second),
        "ci_95": [float(low), float(high)],
        "p_value": float(min(1.0, 2 * min(
            (differences <= 0).mean(), (differences >= 0).mean()
        ))),
    }


def main():
    reviews = {name: load_review(path) for name, path in REVIEW_PATHS.items()}
    names = list(reviews)
    shared = sorted(set(reviews[names[0]]) & set(reviews[names[1]]))

    with NLI_PATH.open(encoding="utf-8") as f:
        nli = list(csv.DictReader(f))

    # A paraphrase counts as good only if no reviewer said "no".
    good = {}

    for row in nli:
        votes = [r[row["item_id"]] for r in reviews.values()
                 if row["item_id"] in r]
        good[row["item_id"]] = bool(votes) and all(votes)

    assert all(
        any(row["item_id"] in r for r in reviews.values()) for row in nli
    ), "some paraphrases were not reviewed"

    with LABELS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "query_id", "original_question", "paraphrase",
            *[f"same_meaning_{n}" for n in names], "same_meaning_combined",
        ])

        for row in nli:
            writer.writerow([
                row["item_id"], row["query_id"], row["original_question"],
                row["paraphrase"],
                *[("yes" if reviews[n][row["item_id"]] else "no")
                  if row["item_id"] in reviews[n] else "" for n in names],
                "yes" if good[row["item_id"]] else "no",
            ])

    query_ids = sorted({row["query_id"] for row in nli})
    good_count = {
        q: sum(good[f"{q}_{p}"] for p in PARAPHRASES) for q in query_ids
    }

    summary = {
        "reviewers": {
            n: {"rows": len(r), "yes_rate": float(np.mean(list(r.values())))}
            for n, r in reviews.items()
        },
        "between_reviewers": {
            **agreement(
                [reviews[names[0]][i] for i in shared],
                [reviews[names[1]][i] for i in shared],
            ),
            "disagreements": [
                i for i in shared
                if reviews[names[0]][i] != reviews[names[1]][i]
            ],
        },
        "combined": {
            "paraphrases": len(good),
            "good": int(sum(good.values())),
            "good_rate": float(np.mean(list(good.values()))),
            "questions_by_good_paraphrases": {
                str(k): sum(v == k for v in good_count.values())
                for k in range(len(PARAPHRASES) + 1)
            },
        },
        "nli_vs_combined_review": agreement(
            [row["nli_same_meaning"] == "yes" for row in nli],
            [good[row["item_id"]] for row in nli],
        ),
    }

    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    attacks = np.array([r["attack"] for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    z_original = np.array([float(r["z_original"]) for r in rows])
    z_paraphrases = np.array([
        [float(r[f"z_{p}"]) for p in PARAPHRASES] for r in rows
    ])
    is_good = np.array([
        [good[f"{r['query_id']}_{p}"] for p in PARAPHRASES] for r in rows
    ])

    all_drop = drop(z_original, z_paraphrases, np.ones_like(is_good))
    rng = np.random.default_rng(SEED)

    for attack in ATTACKS:
        in_attack = (attacks == "none") | (attacks == attack)
        labels = (attacks == attack).astype(int)
        has_good = is_good.any(axis=1)
        has_both = has_good & ~is_good.all(axis=1)

        good_drop = drop(z_original, z_paraphrases, is_good)
        bad_drop = drop(z_original, z_paraphrases, ~is_good)

        # 1. Good paraphrases only against all three, on the questions
        #    that have at least one good paraphrase.
        keep = in_attack & has_good
        good_vs_all = compare(
            labels[keep], groups[keep], good_drop[keep], all_drop[keep], rng
        )

        # 2. Within the same questions: good paraphrases against bad ones.
        keep = in_attack & has_both
        good_vs_bad = compare(
            labels[keep], groups[keep], good_drop[keep], bad_drop[keep], rng
        )

        summary[attack] = {
            "good_only_vs_all_three": good_vs_all,
            "good_vs_bad_same_questions": good_vs_bad,
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nMANUAL PARAPHRASE REVIEW")
    print("=" * 66)

    for n, r in summary["reviewers"].items():
        print(f"{n}: {r['rows']} rows, yes rate {r['yes_rate']:.1%}")

    b = summary["between_reviewers"]
    print(f"Shared rows: {b['rows']}, agreement {b['agreement']:.1%}, "
          f"Cohen's kappa {b['cohen_kappa']:.3f}")
    print("Disagreements:", ", ".join(b["disagreements"]) or "none")

    c = summary["combined"]
    print(f"Good paraphrases (no reviewer said no): {c['good']}/"
          f"{c['paraphrases']} ({c['good_rate']:.1%})")
    print("Questions by number of good paraphrases:",
          c["questions_by_good_paraphrases"])

    n = summary["nli_vs_combined_review"]
    print(f"NLI check against combined review: agreement "
          f"{n['agreement']:.1%}, kappa {n['cohen_kappa']:.3f}")

    print("\nRELEVANCE CONSISTENCY (z_drop) BY PARAPHRASE QUALITY")
    print("=" * 66)

    for attack in ATTACKS:
        for key, first, second in (
            ("good_only_vs_all_three", "good only", "all three"),
            ("good_vs_bad_same_questions", "good", "bad"),
        ):
            s = summary[attack][key]
            print(f"\n{attack}: {first} vs {second} "
                  f"({s['questions']} questions, {s['poisoned']} poisoned, "
                  f"{s['benign']} benign)")
            print(f"  AUROC {s['auroc_first']:.3f} vs "
                  f"{s['auroc_second']:.3f}, difference "
                  f"{s['auroc_difference']:+.3f} "
                  f"[{s['ci_95'][0]:+.3f}, {s['ci_95'][1]:+.3f}], "
                  f"p={s['p_value']:.3f}")
            print(f"  TPR@5%FPR {s['tpr_at_5pct_fpr_first']:.3f} vs "
                  f"{s['tpr_at_5pct_fpr_second']:.3f}")

    print("\nCombined labels:", LABELS_PATH)
    print("Summary saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
