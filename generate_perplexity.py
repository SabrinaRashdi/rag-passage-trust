
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Perplexity baseline: GPT-2, as used for perplexity-based detection in
# PoisonedRAG and RAGuard.
MODEL_NAME = "gpt2"
TRUST_DIR = Path("data_processed/trust")

EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
OUTPUT_PATH = TRUST_DIR / "perplexity.csv"

MAX_LENGTH = 512


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading GPT-2 on", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    print("Passages loaded:", len(passages))

    rows = []

    with torch.inference_mode():
        for i, passage in enumerate(passages, 1):
            inputs = tokenizer(
                passage["text"],
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            # Mean negative log-likelihood per token.
            loss = model(**inputs, labels=inputs["input_ids"]).loss

            rows.append({
                "passage_id": passage["passage_id"],
                "log_perplexity": float(loss),
            })

            if i % 500 == 0:
                print(f"Processed {i}/{len(passages)} passages", flush=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nPERPLEXITY RESULTS")
    print("=" * 45)
    print("Passages scored:", len(rows))
    print("Output file:", OUTPUT_PATH)
    print("Perplexity generation: SUCCESS")


if __name__ == "__main__":
    main()
