
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data_processed")
RETRIEVAL_PATH = DATA_DIR / "experiment_retrieval_results.json"
MAPPING_PATH = DATA_DIR / "experiment_query_mapping.json"
QRELS_PATH = Path("datasets/nq/qrels/test.tsv")
OUTPUT_PATH = DATA_DIR / "paraphrase_robustness_analysis.json"


def main():
    with RETRIEVAL_PATH.open(encoding="utf-8") as f:
        retrieval = json.load(f)

    with MAPPING_PATH.open(encoding="utf-8") as f:
        mapping = json.load(f)

    relevant = {}

    with QRELS_PATH.open(encoding="utf-8") as f:
        next(f)  # Skip the header row

        for line in f:
            parts = line.strip().split("\t")

            if len(parts) == 4:
                qid, _, docid, score = parts
            elif len(parts) == 3:
                qid, docid, score = parts
            else:
                raise ValueError(f"Unexpected qrels row: {line!r}")

            if int(score) > 0:
                relevant.setdefault(qid, set()).add(docid)

    grouped = {}

    for experiment_id, ranked in retrieval.items():
        info = mapping[experiment_id]
        source_id = info["source_id"]
        query_type = info["type"]

        gold = relevant.get(source_id, set())
        ranked_ids = list(ranked.keys())

        hit10 = int(any(
            pid in gold for pid in ranked_ids[:10]
        ))

        hit100 = int(any(
            pid in gold for pid in ranked_ids[:100]
        ))

        mrr10 = 0.0

        for rank, pid in enumerate(ranked_ids[:10], 1):
            if pid in gold:
                mrr10 = 1.0 / rank
                break

        grouped.setdefault(source_id, {})[experiment_id] = {
            "type": query_type,
            "hit10": hit10,
            "hit100": hit100,
            "mrr10": mrr10,
        }

    rows = []

    for source_id, variants in grouped.items():
        originals = [
            v for v in variants.values()
            if v["type"] == "original"
        ]

        paraphrases = [
            v for v in variants.values()
            if v["type"] == "paraphrase"
        ]

        if len(originals) != 1 or len(paraphrases) != 3:
            raise ValueError(
                f"Unexpected variant counts for {source_id}"
            )

        original = originals[0]

        row = {"source_id": source_id}

        for metric in ["hit10", "hit100", "mrr10"]:
            paraphrase_mean = float(np.mean([
                p[metric] for p in paraphrases
            ]))

            row[metric] = {
                "original": original[metric],
                "paraphrase_mean": paraphrase_mean,
                "difference": paraphrase_mean - original[metric],
            }

        rows.append(row)

    summary = {}

    for metric in ["hit10", "hit100", "mrr10"]:
        differences = np.array([
            row[metric]["difference"] for row in rows
        ])

        summary[metric] = {
            "improved": int(np.sum(differences > 0)),
            "worsened": int(np.sum(differences < 0)),
            "unchanged": int(np.sum(differences == 0)),
            "mean_difference": float(np.mean(differences)),
        }

    output = {
        "questions_analysed": len(rows),
        "paraphrases_per_question": 3,
        "summary": summary,
        "per_question": rows,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nPARAPHRASE ROBUSTNESS ANALYSIS")
    print("=" * 45)
    print("Questions analysed:", len(rows))

    for metric, values in summary.items():
        print(f"\nMetric: {metric}")
        for key, value in values.items():
            print(f"{key}: {value}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
