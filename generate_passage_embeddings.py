
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from src.contriever_src.contriever import Contriever

MODEL_NAME = "facebook/contriever"
DATA_DIR = Path("data_processed")

INPUT_PATH = DATA_DIR / "candidate_passages.jsonl"
OUTPUT_PATH = DATA_DIR / "candidate_embeddings.npy"
IDS_PATH = DATA_DIR / "candidate_embedding_ids.json"

BATCH_SIZE = 8
MAX_LENGTH = 128


def main():
    print("Loading Contriever model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Contriever.from_pretrained(MODEL_NAME)
    model = model.to("cpu")
    model.eval()

    passage_ids = []
    passage_texts = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            passage_ids.append(str(record["_id"]))

            title = record.get("title", "")
            text = record.get("text", "")

            passage_texts.append(
                (title + " " + text).strip()
            )

    print(f"Passages loaded: {len(passage_texts)}")

    all_embeddings = []

    with torch.inference_mode():
        for start in range(0, len(passage_texts), BATCH_SIZE):
            batch = passage_texts[start:start + BATCH_SIZE]

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

            if (start // BATCH_SIZE + 1) % 100 == 0:
                print(
                    f"Processed {min(start + BATCH_SIZE, len(passage_texts))}"
                    f"/{len(passage_texts)} passages",
                    flush=True,
                )

    embeddings = np.concatenate(all_embeddings, axis=0)

    np.save(OUTPUT_PATH, embeddings)

    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(passage_ids, f, indent=2)

    print("\nPASSAGE EMBEDDING RESULTS")
    print("=" * 45)
    print("Total passages:", len(passage_ids))
    print("Embedding shape:", embeddings.shape)
    print("NaN values:", np.isnan(embeddings).any())
    print("Embedding generation: SUCCESS")


if __name__ == "__main__":
    main()
