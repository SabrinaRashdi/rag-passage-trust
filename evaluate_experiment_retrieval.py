import json
from pathlib import Path

DATA = Path("data_processed")
RESULTS = DATA / "experiment_retrieval_results.json"
MAPPING = DATA / "experiment_query_mapping.json"
OUTPUT = DATA / "experiment_retrieval_metrics.json"
QRELS = Path("datasets/nq/qrels/test.tsv")

with RESULTS.open(encoding="utf-8") as f:
    retrieval = json.load(f)

with MAPPING.open(encoding="utf-8") as f:
    mapping = json.load(f)

relevant = {}

with QRELS.open(encoding="utf-8") as f:
    next(f)
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

groups = {"original": [], "paraphrase": []}

assert set(retrieval) == set(mapping), "Query IDs do not match"

for query_id, results in retrieval.items():
    info = mapping[query_id]
    gold = relevant[info["source_id"]]
    ranked = list(results.keys())

    row = {}

    for k in [1, 5, 10, 100]:
        row[f"hit@{k}"] = float(
            any(doc in gold for doc in ranked[:k])
        )

    for k in [10, 100]:
        row[f"recall@{k}"] = (
            sum(doc in gold for doc in ranked[:k]) / len(gold)
        )

    row["mrr@10"] = next(
        (
            1.0 / rank
            for rank, doc in enumerate(ranked[:10], 1)
            if doc in gold
        ),
        0.0
    )

    groups[info["type"]].append(row)

metrics = {}

for group, rows in groups.items():
    metrics[group] = {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in rows[0]
    }

output = {
    "evaluation_scope": "restricted_candidate_pool",
    "candidate_passages": 9872,
    "max_length": 512,
    "total_queries": len(retrieval),
    "group_counts": {
        group: len(rows) for group, rows in groups.items()
    },
    "metrics": metrics
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\n512-TOKEN RETRIEVAL METRICS")
print("=" * 45)

for group, values in metrics.items():
    print(f"\n{group.upper()} ({len(groups[group])} queries)")
    for metric, value in values.items():
        print(f"{metric}: {value:.6f}")

print("\nSaved to:", OUTPUT)
