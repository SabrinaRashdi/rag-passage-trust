
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from src.contriever_src.contriever import Contriever

MODEL_NAME = "facebook/contriever"
DATA_DIR = Path("data_processed")

INPUT_PATH = DATA_DIR / "experiment_queries.json"
OUTPUT_PATH = DATA_DIR / "query_embeddings.npy"
IDS_PATH = DATA_DIR / "query_embedding_ids.json"

BATCH_SIZE = 8
MAX_LENGTH = 128


def main():
    print("Loading experiment queries...")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print("Queries loaded:", len(query_texts))

    print("Loading Contriever model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Contriever.from_pretrained(MODEL_NAME)
    model = model.to("cpu")
    model.eval()

    all_embeddings = []

    with torch.inference_mode():
        for start in range(0, len(query_texts), BATCH_SIZE):
            batch = query_texts[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            embeddings = model(**inputs)

            all_embeddings.append(
                embeddings.cpu().numpy().astype("float32")
            )

            if (start // BATCH_SIZE + 1) % 10 == 0:
                print(
                    f"Processed {min(start + BATCH_SIZE, len(query_texts))}"
                    f"/{len(query_texts)} queries",
                    flush=True,
                )

    embeddings = np.concatenate(all_embeddings, axis=0)

    np.save(OUTPUT_PATH, embeddings)

    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(query_ids, f, indent=2)

    print("\nQUERY EMBEDDING RESULTS")
    print("=" * 45)
    print("Total queries:", len(query_ids))
    print("Embedding shape:", embeddings.shape)
    print("NaN values:", np.isnan(embeddings).any())
    print("Query embedding generation: SUCCESS")


if __name__ == "__main__":
    main()
