
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Extractive reader trained on SQuAD 2.0, so it can also say "no answer".
MODEL_NAME = "deepset/roberta-base-squad2"
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
OUTPUT_PATH = TRUST_DIR / "answer_support.csv"

VARIANTS = ("original", "para1", "para2", "para3")

BATCH_SIZE = 16
MAX_LENGTH = 512
MAX_ANSWER_TOKENS = 30
N_BEST = 20


def read(questions, passages, tokenizer, model, device):
    """Best answer span and support score for each (question, passage)."""
    inputs = tokenizer(
        questions,
        passages,
        padding=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = inputs.pop("offset_mapping")
    sequence_ids = [inputs.sequence_ids(i) for i in range(len(questions))]

    with torch.inference_mode():
        output = model(**inputs.to(device))

    start_logits = output.start_logits.cpu()
    end_logits = output.end_logits.cpu()

    results = []

    for i in range(len(questions)):
        # Score of the "no answer" option (the first token).
        null_score = float(start_logits[i, 0] + end_logits[i, 0])

        in_passage = [
            j for j, s in enumerate(sequence_ids[i]) if s == 1
        ]
        first, last = in_passage[0], in_passage[-1]

        starts = torch.topk(
            start_logits[i, first:last + 1],
            min(N_BEST, last - first + 1)
        ).indices + first
        ends = torch.topk(
            end_logits[i, first:last + 1],
            min(N_BEST, last - first + 1)
        ).indices + first

        best_score, best_span = None, None

        for s in starts.tolist():
            for e in ends.tolist():
                if e < s or e - s + 1 > MAX_ANSWER_TOKENS:
                    continue

                score = float(start_logits[i, s] + end_logits[i, e])

                if best_score is None or score > best_score:
                    best_score, best_span = score, (s, e)

        s, e = best_span
        answer = passages[i][int(offsets[i, s, 0]):int(offsets[i, e, 1])]

        results.append({
            "answer": answer.strip(),
            # Positive: the reader believes the passage answers the question.
            "support": best_score - null_score,
        })

    return results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading reader model on", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    # One job per (passage, query variant) pair.
    jobs = [
        (p, variant)
        for p in passages
        for variant in VARIANTS
    ]
    # Process similar lengths together so batches need less padding.
    jobs.sort(key=lambda job: len(job[0]["text"]))

    print("Passages:", len(passages))
    print("Passage-question pairs:", len(jobs))

    rows = {
        p["passage_id"]: {
            "passage_id": p["passage_id"],
            "query_id": p["query_id"],
        }
        for p in passages
    }

    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]

        outputs = read(
            [queries[f"{p['query_id']}_{v}"] for p, v in batch],
            [p["text"] for p, _ in batch],
            tokenizer, model, device,
        )

        for (p, variant), out in zip(batch, outputs):
            rows[p["passage_id"]][f"answer_{variant}"] = out["answer"]
            rows[p["passage_id"]][f"support_{variant}"] = out["support"]

        if (start // BATCH_SIZE + 1) % 50 == 0:
            print(
                f"Processed {min(start + BATCH_SIZE, len(jobs))}"
                f"/{len(jobs)} pairs",
                flush=True,
            )

    fields = ["passage_id", "query_id"]
    fields += [f"answer_{v}" for v in VARIANTS]
    fields += [f"support_{v}" for v in VARIANTS]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[p["passage_id"]] for p in passages)

    print("\nANSWER SUPPORT RESULTS")
    print("=" * 45)
    print("Passages scored:", len(rows))
    print("Output file:", OUTPUT_PATH)
    print("Answer support generation: SUCCESS")


if __name__ == "__main__":
    main()
