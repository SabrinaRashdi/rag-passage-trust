import json
from pathlib import Path

DATA = Path("data_processed")
CORPUS = Path("datasets/nq/corpus.jsonl")
IDS_FILE = DATA / "candidate_passage_ids.json"
ORIGINAL = DATA / "candidate_passages.jsonl"
OUTPUT = DATA / "candidate_passages_reconstructed.jsonl"

with IDS_FILE.open(encoding="utf-8") as f:
    selected_ids = set(map(str, json.load(f)))

print("Selected passage IDs:", len(selected_ids))

found = {}

with CORPUS.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        doc_id = str(record["_id"])

        if doc_id in selected_ids:
            found[doc_id] = record

if set(found) != selected_ids:
    missing = selected_ids - set(found)
    raise ValueError(f"Missing passages: {len(missing)}")

# Preserve the original candidate file's ordering.
with ORIGINAL.open(encoding="utf-8") as f:
    original_records = [
        json.loads(line)
        for line in f if line.strip()
    ]

original_ids = [str(r["_id"]) for r in original_records]

if set(original_ids) != selected_ids:
    raise ValueError("Original candidate IDs do not match saved IDs")

with OUTPUT.open("w", encoding="utf-8") as f:
    for doc_id in original_ids:
        f.write(json.dumps(found[doc_id], ensure_ascii=False) + "\n")

print("Reconstructed passages:", len(found))
print("Output:", OUTPUT)

# Verify IDs, ordering, and passage content.
with OUTPUT.open(encoding="utf-8") as f:
    reconstructed = [
        json.loads(line) for line in f if line.strip()
    ]

assert len(reconstructed) == 9872
assert [r["_id"] for r in reconstructed] == original_ids

content_matches = all(
    a == b for a, b in zip(reconstructed, original_records)
)

print("IDs and ordering match: True")
print("Full records match:", content_matches)

if not content_matches:
    raise ValueError("Reconstructed content differs from original")

print("RECONSTRUCTION SUCCESSFUL")
