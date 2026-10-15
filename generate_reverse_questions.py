
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generate_trust_embeddings import MODEL_NAME as CONTRIEVER_NAME
from generate_trust_embeddings import encode
from src.contriever_src.contriever import Contriever

# Reverse questions (idea 1): for every evaluated passage, a doc2query
# model writes the questions that the passage answers. A poisoned passage
# is written to answer one question, so its questions should all be close
# to the user's question; a real passage usually answers several.
# The generated questions are then encoded with Contriever, and two
# signals are saved for every passage (see question_signals).
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
ADAPTIVE_PATH = TRUST_DIR / "adaptive_attack_passages.jsonl"
QUESTIONS_PATH = TRUST_DIR / "reverse_questions.csv"
SCORES_PATH = TRUST_DIR / "reverse_question_scores.csv"
EMBEDDING_NAME = "reverse_question"

# doc2query T5 model from BEIR, trained on MS MARCO.
GENERATOR_NAME = "BeIR/query-gen-msmarco-t5-base-v1"
N_QUESTIONS = 5
# Sampling settings from the BEIR query-generation example.
TOP_P = 0.95
MAX_INPUT_LENGTH = 384
MAX_QUESTION_LENGTH = 64
BATCH_SIZE = 16
SEED = 42


def normalise(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def question_signals(generated, user):
    """
    generated: Contriever embeddings of the questions generated from one
    passage; user: embedding of the user's question. Returns
    - question_focus: mean cosine similarity between the generated
      questions and the user's question, and
    - question_sameness: mean cosine similarity between pairs of
      generated questions (needs at least two questions).
    A higher value is more suspicious for both.
    """
    generated = normalise(generated)
    focus = float((generated @ normalise(user)).mean())

    if len(generated) < 2:
        return focus, float("nan")

    pairs = generated @ generated.T
    return focus, float(pairs[np.triu_indices(len(generated), k=1)].mean())


def main():
    passages = []

    for path in (EVAL_PATH, ADAPTIVE_PATH):
        with path.open(encoding="utf-8") as f:
            passages.extend(json.loads(line) for line in f)

    # The reworded attack file has no benign passages, so no passage is
    # in both files.
    passages.sort(key=lambda p: p["passage_id"])
    assert len({p["passage_id"] for p in passages}) == len(passages)

    # Questions are saved after each batch, so a stopped run continues
    # where it stopped (batches and seeds stay the same).
    questions = {}

    if QUESTIONS_PATH.exists():
        with QUESTIONS_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                questions[row["passage_id"]] = [
                    row[f"question_{i + 1}"] for i in range(N_QUESTIONS)
                ]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    new_file = not QUESTIONS_PATH.exists()

    if len(questions) < len(passages):
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_NAME)
        model = model.to(device).eval()

    with QUESTIONS_PATH.open("a", newline="", encoding="utf-8") as f, \
            torch.inference_mode():
        writer = csv.writer(f)

        if new_file:
            writer.writerow(["passage_id", *[f"question_{i + 1}"
                                             for i in range(N_QUESTIONS)]])

        for start in range(0, len(passages), BATCH_SIZE):
            batch = passages[start:start + BATCH_SIZE]

            if all(p["passage_id"] in questions for p in batch):
                continue

            inputs = tokenizer(
                [p["text"] for p in batch], padding=True, truncation=True,
                max_length=MAX_INPUT_LENGTH, return_tensors="pt",
            ).to(device)
            # The same seed for each batch, so a run gives the same
            # questions.
            torch.manual_seed(SEED + start)
            output = model.generate(
                **inputs, do_sample=True, top_p=TOP_P,
                max_length=MAX_QUESTION_LENGTH,
                num_return_sequences=N_QUESTIONS,
            )
            texts = tokenizer.batch_decode(output, skip_special_tokens=True)

            for i, p in enumerate(batch):
                questions[p["passage_id"]] = [
                    t.strip() for t in
                    texts[i * N_QUESTIONS:(i + 1) * N_QUESTIONS]
                ]
                writer.writerow([p["passage_id"],
                                 *questions[p["passage_id"]]])

            f.flush()

            if (start // BATCH_SIZE + 1) % 20 == 0:
                print(f"generated {start + len(batch)}/{len(passages)}",
                      flush=True)

    assert len(questions) == len(passages)
    questions = {p["passage_id"]: questions[p["passage_id"]]
                 for p in passages}

    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_NAME)
    model = Contriever.from_pretrained(CONTRIEVER_NAME).to(device).eval()
    ids = [f"{pid}#{i}" for pid in questions for i in range(N_QUESTIONS)]
    embeddings = encode(
        [q for qs in questions.values() for q in qs],
        tokenizer, model, device, EMBEDDING_NAME,
    )
    np.save(TRUST_DIR / f"{EMBEDDING_NAME}_embeddings.npy", embeddings)

    with (TRUST_DIR / f"{EMBEDDING_NAME}_embedding_ids.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(ids, f)

    # The two signals for every passage, against the user's question.
    query_emb = np.load(TRUST_DIR / "query_embeddings.npy")

    with (TRUST_DIR / "query_embedding_ids.json").open(
        encoding="utf-8"
    ) as f:
        query_index = {qid: i for i, qid in enumerate(json.load(f))}

    query_of = {p["passage_id"]: p["query_id"] for p in passages}

    with SCORES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["passage_id", "question_focus", "question_sameness"])

        for n, pid in enumerate(questions):
            focus, sameness = question_signals(
                embeddings[n * N_QUESTIONS:(n + 1) * N_QUESTIONS],
                query_emb[query_index[f"{query_of[pid]}_original"]],
            )
            writer.writerow([pid, focus, sameness])

    print("Passages:", len(questions))
    print("Saved:", QUESTIONS_PATH, "and", EMBEDDING_NAME, "embeddings",
          embeddings.shape)


if __name__ == "__main__":
    main()
