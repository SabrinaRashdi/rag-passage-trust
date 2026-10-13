
import csv
import json
import random
from pathlib import Path

DATA_DIR = Path("data_processed")

PARAPHRASES_PATH = DATA_DIR / "nq_reworded_questions_corrected.json"
VALIDATION_PATH = DATA_DIR / "corrected_paraphrase_validation.csv"

REVIEWERS = ("sanduni", "sabrina")
SHARED_COUNT = 50
SEED = 42

FIELDS = [
    "item_id",
    "query_id",
    "original_question",
    "paraphrase",
    "same_meaning",
    "note",
]


def main():
    with PARAPHRASES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    # Paraphrases flagged by the automatic similarity check (< 0.75).
    with VALIDATION_PATH.open(encoding="utf-8") as f:
        flagged = {
            (row["query_id"], row["reworded_question"])
            for row in csv.DictReader(f)
            if row["status"] == "REVIEW"
        }

    items = []

    for qid, record in data.items():
        for i, paraphrase in enumerate(record["reworded_questions"], 1):
            items.append({
                "item_id": f"{qid}_para{i}",
                "query_id": qid,
                "original_question": record["original_question"],
                "paraphrase": paraphrase,
                "same_meaning": "",
                "note": "",
            })

    flagged_items = [
        item for item in items
        if (item["query_id"], item["paraphrase"]) in flagged
    ]
    other_items = [
        item for item in items
        if (item["query_id"], item["paraphrase"]) not in flagged
    ]

    rng = random.Random(SEED)
    rng.shuffle(other_items)

    # Both reviewers label every flagged paraphrase plus a random sample,
    # so agreement can be measured on the shared items.
    extra_shared = SHARED_COUNT - len(flagged_items)
    shared = flagged_items + other_items[:extra_shared]
    remaining = other_items[extra_shared:]

    half = len(remaining) // 2
    own = {
        REVIEWERS[0]: remaining[:half],
        REVIEWERS[1]: remaining[half:],
    }

    for reviewer in REVIEWERS:
        rows = shared + own[reviewer]
        rng_rows = random.Random(SEED)
        rng_rows.shuffle(rows)

        output_path = DATA_DIR / f"paraphrase_manual_review_{reviewer}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"{reviewer}: {len(rows)} paraphrases -> {output_path}")

    print("\nMANUAL REVIEW FILES")
    print("=" * 45)
    print("Total paraphrases:", len(items))
    print("Flagged by similarity check:", len(flagged_items))
    print("Shared items (both reviewers):", len(shared))
    print("Own items per reviewer:", {r: len(own[r]) for r in REVIEWERS})


if __name__ == "__main__":
    main()
