
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compute_trust_scores import VARIANTS, load_embeddings
from evaluate_detection import ATTACKS, TARGET_FPR, auroc

# Failure-case analysis for the relevance-consistency score (z_drop):
# which poisoned passages are missed, which benign passages are wrongly
# flagged, and what the failures have in common.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

SCORES_PATH = TRUST_DIR / "trust_scores.csv"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"

CASES_PATH = TRUST_DIR / "failure_cases.csv"
SUMMARY_PATH = TRUST_DIR / "failure_analysis.json"

TEXT_PREVIEW_CHARS = 200


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

    query_emb, query_index = load_embeddings("query")

    attacks = np.array([r["attack"] for r in rows])
    query_ids = np.array([r["query_id"] for r in rows])
    is_gold = np.array([r["is_gold"] == "1" for r in rows])
    z_drop = np.array([float(r["z_drop"]) for r in rows])
    z_original = np.array([float(r["z_original"]) for r in rows])

    benign = attacks == "none"
    questions = sorted(set(query_ids))

    # One threshold for everything: 5% of benign passages are flagged.
    threshold = float(np.quantile(z_drop[benign], 1 - TARGET_FPR))
    flagged = z_drop > threshold

    # How close the paraphrases are to the original question (cosine).
    def paraphrase_similarity(qid):
        vectors = np.stack([
            query_emb[query_index[f"{qid}_{v}"]] for v in VARIANTS
        ])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return float((vectors[1:] @ vectors[0]).mean())

    similarity = np.array([paraphrase_similarity(q) for q in questions])
    question_words = np.array([
        len(queries[f"{q}_original"].split()) for q in questions
    ])

    def per_question(values, mask, function=np.mean):
        return np.array([
            function(values[mask & (query_ids == q)]) for q in questions
        ])

    false_alarms = per_question(flagged, benign, np.sum)
    benign_median_drop = per_question(z_drop, benign, np.median)

    low = similarity < np.quantile(similarity, 0.25)
    high = similarity >= np.quantile(similarity, 0.75)

    def rate_in(question_mask, passage_mask):
        chosen = np.isin(query_ids, np.array(questions)[question_mask])
        return float(flagged[chosen & passage_mask].mean())

    summary = {
        "threshold_z_drop": threshold,
        "false_alarms": {
            "benign_flagged": int((flagged & benign).sum()),
            "gold_flagged": int((flagged & benign & is_gold).sum()),
            "gold_total": int(is_gold.sum()),
            "questions_with_false_alarm": int((false_alarms > 0).sum()),
            "most_in_one_question": int(false_alarms.max()),
            "false_alarms_vs_paraphrase_similarity": correlation(
                similarity, false_alarms
            ),
            "fpr_low_similarity_quarter": rate_in(low, benign),
            "fpr_high_similarity_quarter": rate_in(high, benign),
        },
    }

    for attack in ATTACKS:
        poisoned = attacks == attack
        caught = per_question(flagged, poisoned)
        mean_drop = per_question(z_drop, poisoned)

        # Post-hoc variant (found by this analysis, not planned before):
        # z_drop relative to the median of the question's candidate set,
        # which removes the part of the drop shared by all passages when
        # a paraphrase drifts. The median uses no labels.
        in_set = benign | poisoned
        set_median = dict(zip(
            questions, per_question(z_drop, in_set, np.median)
        ))
        relative = z_drop - np.array([set_median[q] for q in query_ids])
        relative_threshold = np.quantile(relative[benign], 1 - TARGET_FPR)
        labels = poisoned[in_set].astype(int)

        summary[attack] = {
            "tpr_at_5pct_fpr": float(flagged[poisoned].mean()),
            "questions_all_caught": int((caught == 1).sum()),
            "questions_all_missed": int((caught == 0).sum()),
            "questions_mixed": int(((caught > 0) & (caught < 1)).sum()),
            "tpr_low_similarity_quarter": rate_in(low, poisoned),
            "tpr_high_similarity_quarter": rate_in(high, poisoned),
            "poisoned_drop_vs_paraphrase_similarity": correlation(
                similarity, mean_drop
            ),
            "poisoned_drop_vs_question_words": correlation(
                question_words, mean_drop
            ),
            "poisoned_drop_vs_original_z": correlation(
                per_question(z_original, poisoned), mean_drop
            ),
            "poisoned_drop_vs_benign_median_drop": correlation(
                benign_median_drop, mean_drop
            ),
            "post_hoc_set_relative_z_drop": {
                "auroc_z_drop": auroc(labels, z_drop[in_set]),
                "auroc_set_relative": auroc(labels, relative[in_set]),
                "tpr_at_5pct_fpr_set_relative": float(
                    (relative[poisoned] > relative_threshold).mean()
                ),
            },
        }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Every wrong decision, with the question wordings, for reading by hand.
    cases = []

    for i, row in enumerate(rows):
        if row["attack"] == "none" and flagged[i]:
            error = "false_alarm"
        elif row["attack"] != "none" and not flagged[i]:
            error = "missed"
        else:
            continue

        qid = row["query_id"]
        case = {
            "error": error,
            "attack": row["attack"],
            "passage_id": row["passage_id"],
            "query_id": qid,
            "is_gold": row["is_gold"],
            "z_drop": round(z_drop[i], 3),
            "z_original": round(z_original[i], 3),
            "paraphrase_similarity": round(
                similarity[questions.index(qid)], 3
            ),
        }

        for variant in VARIANTS:
            case[f"question_{variant}"] = queries[f"{qid}_{variant}"]

        case["passage_text"] = texts[row["passage_id"]][:TEXT_PREVIEW_CHARS]
        cases.append(case)

    cases.sort(key=lambda c: (c["error"], c["attack"], c["query_id"]))

    with CASES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)

    print("\nFAILURE-CASE ANALYSIS (z_drop, threshold at 5% false alarms)")
    print("=" * 70)
    print(f"Threshold on z_drop: {threshold:.3f}")

    alarms = summary["false_alarms"]
    print(f"\nFalse alarms: {alarms['benign_flagged']} benign passages "
          f"in {alarms['questions_with_false_alarm']} questions "
          f"(most in one question: {alarms['most_in_one_question']})")
    print(f"Gold passages flagged: {alarms['gold_flagged']} "
          f"of {alarms['gold_total']}")
    print("False-alarm rate, least similar paraphrases: "
          f"{alarms['fpr_low_similarity_quarter']:.3f}")
    print("False-alarm rate, most similar paraphrases:  "
          f"{alarms['fpr_high_similarity_quarter']:.3f}")

    for attack in ATTACKS:
        s = summary[attack]
        post_hoc = s["post_hoc_set_relative_z_drop"]
        print(f"\nAttack: {attack}")
        print(f"  Poisoned passages caught: {s['tpr_at_5pct_fpr']:.3f}")
        print(f"  Questions all caught / mixed / all missed: "
              f"{s['questions_all_caught']} / {s['questions_mixed']} / "
              f"{s['questions_all_missed']}")
        print("  Caught, least / most similar paraphrases: "
              f"{s['tpr_low_similarity_quarter']:.3f} / "
              f"{s['tpr_high_similarity_quarter']:.3f}")

        for key in ("poisoned_drop_vs_original_z",
                    "poisoned_drop_vs_benign_median_drop",
                    "poisoned_drop_vs_question_words",
                    "poisoned_drop_vs_paraphrase_similarity"):
            print(f"  {key}: rho={s[key]['spearman']:+.3f} "
                  f"(p={s[key]['p_value']:.4f})")

        print(f"  Post hoc, set-relative z_drop: AUROC "
              f"{post_hoc['auroc_z_drop']:.3f} -> "
              f"{post_hoc['auroc_set_relative']:.3f}, caught "
              f"{post_hoc['tpr_at_5pct_fpr_set_relative']:.3f}")

    print("\nCases written:", len(cases), "->", CASES_PATH)
    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
