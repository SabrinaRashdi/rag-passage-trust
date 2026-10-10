
import json
import csv
from sentence_transformers import SentenceTransformer, util

INPUT_PATH = "data_processed/nq_reworded_questions.json"
OUTPUT_PATH = "data_processed/paraphrase_validation.csv"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for qid, record in data.items():
    original = record["original_question"]
    rewrites = record["reworded_questions"]

    sentences = [original] + rewrites
    embeddings = model.encode(
        sentences,
        convert_to_tensor=True
    )

    for i, rewrite in enumerate(rewrites):
        similarity = util.cos_sim(
            embeddings[0],
            embeddings[i + 1]
        ).item()

        status = (
            "REVIEW" if similarity < 0.80
            else "PASS"
        )

        rows.append({
            "query_id": qid,
            "original_question": original,
            "reworded_question": rewrite,
            "similarity": round(similarity, 4),
            "status": status
        })

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

review = [r for r in rows if r["status"] == "REVIEW"]

print("Total paraphrases:", len(rows))
print("Flagged for review:", len(review))
print("Not flagged:", len(rows) - len(review))
print("Validation report saved to:", OUTPUT_PATH)

print("\nLowest similarity examples:")

for row in sorted(rows, key=lambda x: x["similarity"])[:10]:
    print("\nQuery ID:", row["query_id"])
    print("Original:", row["original_question"])
    print("Rewrite:", row["reworded_question"])
    print("Similarity:", row["similarity"])
