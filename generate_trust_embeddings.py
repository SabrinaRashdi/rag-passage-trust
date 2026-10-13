
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from src.contriever_src.contriever import Contriever

MODEL_NAME = "facebook/contriever"
DATA_DIR = Path("data_processed")
OUTPUT_DIR = DATA_DIR / "trust"

POOL_PATH = DATA_DIR / "candidate_passages.jsonl"
EVAL_PATH = OUTPUT_DIR / "eval_passages.jsonl"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"

BATCH_SIZE = 16
# Same limit as the Contriever BEIR encoder used by PoisonedRAG.
MAX_LENGTH = 512


def encode(texts, tokenizer, model, device, name):
    # Encode longest texts together so batches need less padding.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    embeddings = np.zeros((len(texts), 768), dtype="float32")

    with torch.inference_mode():
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]

            inputs = tokenizer(
                [texts[i] for i in indices],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            output = model(**inputs)
            embeddings[indices] = output.cpu().numpy().astype("float32")

            if (start // BATCH_SIZE + 1) % 100 == 0:
                print(
                    f"{name}: processed "
                    f"{min(start + BATCH_SIZE, len(order))}/{len(order)}",
                    flush=True,
                )

    assert np.isfinite(embeddings).all()
    return embeddings


def save(name, ids, embeddings):
    np.save(OUTPUT_DIR / f"{name}_embeddings.npy", embeddings)

    with (OUTPUT_DIR / f"{name}_embedding_ids.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(ids, f)

    print(f"{name}: saved embeddings with shape {embeddings.shape}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading Contriever model on", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Contriever.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 1. Clean candidate pool
    pool_ids, pool_texts = [], []

    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            title = record.get("title", "")

            pool_ids.append(str(record["_id"]))
            pool_texts.append(
                title + " " + record["text"] if len(title) > 0
                else record["text"]
            )

    save("pool", pool_ids,
         encode(pool_texts, tokenizer, model, device, "pool"))

    # 2. Poisoned passages (benign passages are already in the pool)
    adv_ids, adv_texts = [], []

    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["label"] == 1:
                adv_ids.append(record["passage_id"])
                adv_texts.append(record["text"])

    save("poisoned", adv_ids,
         encode(adv_texts, tokenizer, model, device, "poisoned"))

    # 3. Original questions and paraphrases
    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    query_ids = list(queries.keys())

    save("query", query_ids,
         encode([queries[q] for q in query_ids],
                tokenizer, model, device, "query"))

    print("\nEmbedding generation: SUCCESS")


if __name__ == "__main__":
    main()
