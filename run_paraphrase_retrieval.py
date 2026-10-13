
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data_processed")

PASSAGE_EMBEDDINGS = DATA_DIR / "candidate_embeddings.npy"
QUERY_EMBEDDINGS = DATA_DIR / "query_embeddings.npy"

PASSAGE_IDS = DATA_DIR / "candidate_embedding_ids.json"
QUERY_IDS = DATA_DIR / "query_embedding_ids.json"

OUTPUT_PATH = DATA_DIR / "experiment_retrieval_results.json"

TOP_K = 100


def main():
    print("Loading embeddings and IDs...")

    passage_embeddings = np.load(PASSAGE_EMBEDDINGS)
    query_embeddings = np.load(QUERY_EMBEDDINGS)

    with PASSAGE_IDS.open("r", encoding="utf-8") as f:
        passage_ids = json.load(f)

    with QUERY_IDS.open("r", encoding="utf-8") as f:
        query_ids = json.load(f)

    assert len(passage_ids) == len(passage_embeddings)
    assert len(query_ids) == len(query_embeddings)
    assert np.isfinite(passage_embeddings).all()
    assert np.isfinite(query_embeddings).all()

    print("Passage embeddings:", passage_embeddings.shape)
    print("Query embeddings:", query_embeddings.shape)

    # Dot-product retrieval, matching the original experiment.
    scores = query_embeddings @ passage_embeddings.T

    results = {}

    for i, qid in enumerate(query_ids):
        row = scores[i]

        top_indices = np.argsort(-row)[:TOP_K]

        results[qid] = {
            passage_ids[j]: float(row[j])
            for j in top_indices
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    print("\nRETRIEVAL RESULTS")
    print("=" * 45)
    print("Queries processed:", len(results))
    print("Candidate passages:", len(passage_ids))
    print("Top passages per query:", TOP_K)
    print(
        "Total ranked query-passage pairs:",
        sum(len(v) for v in results.values())
    )
    print("Output file:", OUTPUT_PATH)
    print("Retrieval: SUCCESS")


if __name__ == "__main__":
    main()
