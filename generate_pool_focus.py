
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from compute_trust_scores import load_embeddings
from generate_reverse_questions import (
    N_QUESTIONS, generate_questions, question_signals,
)
from generate_trust_embeddings import MODEL_NAME as CONTRIEVER_NAME
from generate_trust_embeddings import ENCODE_DEVICE, POOL_PATH, encode
from src.contriever_src.contriever import Contriever

# Set-relative question focus with a reference outside the read list
# (post hoc, fixed before running). Inside the retrieved top 10, half of
# the passages can be poisoned, so their median focus is a poisoned
# passage's focus. Here the reference is a longer list: the top
# REFERENCE_K passages retrieved for the question, which 5 poisoned
# passages cannot take over. This script saves the question focus of each
# question's top REFERENCE_K clean pool passages; the reference of an
# attacked list also contains the poisoned passages that rank in its top
# REFERENCE_K (see reference_median).
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUESTIONS_PATH = TRUST_DIR / "pool_reverse_questions.csv"
OUTPUT_PATH = TRUST_DIR / "pool_question_focus.csv"

REFERENCE_K = 30


def load_pool_focus():
    """{query_id: [(retrieval score, question focus)]} of the top
    REFERENCE_K clean passages of each question."""
    pool = {}

    with OUTPUT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pool.setdefault(row["query_id"], []).append(
                (float(row["retrieval_score"]), float(row["question_focus"]))
            )

    return pool


def reference_median(clean, poisoned):
    """Median focus of the top REFERENCE_K passages among the clean list
    and the poisoned passages, both as (retrieval score, focus) pairs."""
    ranked = sorted([*clean, *poisoned], key=lambda pair: -pair[0])
    return float(np.median([focus for _, focus in ranked[:REFERENCE_K]]))


def main():
    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    query_emb, query_index = load_embeddings("query")
    pool_emb, pool_index = load_embeddings("pool")
    pool_ids = sorted(pool_index, key=pool_index.get)
    questions = sorted({p["query_id"] for p in passages})

    # The top REFERENCE_K pool passages for each original question.
    top = {}

    for qid in questions:
        scores = pool_emb @ query_emb[query_index[f"{qid}_original"]]
        order = np.argsort(-scores)[:REFERENCE_K]
        top[qid] = [(pool_ids[i], float(scores[i])) for i in order]

    # Questions already generated for the evaluated benign passages are
    # reused (they depend only on the passage text).
    known = {}
    question_emb, question_index = load_embeddings("reverse_question")

    for p in passages:
        if p["label"] == 0:
            known[p["doc_id"]] = np.stack([
                question_emb[question_index[f"{p['passage_id']}#{i}"]]
                for i in range(N_QUESTIONS)
            ])

    texts = {}

    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            title = record.get("title", "")
            texts[str(record["_id"])] = (
                title + " " + record["text"] if len(title) > 0
                else record["text"]
            )

    missing = sorted({doc for qid in questions for doc, _ in top[qid]}
                     - set(known))
    print(f"Pool passages in a top {REFERENCE_K}: "
          f"{len({d for q in questions for d, _ in top[q]})}, "
          f"new: {len(missing)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generated = generate_questions(
        [{"passage_id": doc, "text": texts[doc]} for doc in missing],
        QUESTIONS_PATH, device,
    )

    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_NAME)
    model = Contriever.from_pretrained(CONTRIEVER_NAME).eval()
    embeddings = encode(
        [q for doc in missing for q in generated[doc]],
        tokenizer, model, ENCODE_DEVICE, "pool questions",
    )

    for n, doc in enumerate(missing):
        known[doc] = embeddings[n * N_QUESTIONS:(n + 1) * N_QUESTIONS]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "doc_id", "rank", "retrieval_score",
                         "question_focus"])

        for qid in questions:
            user = query_emb[query_index[f"{qid}_original"]]

            for rank, (doc, score) in enumerate(top[qid], 1):
                focus, _ = question_signals(known[doc], user)
                writer.writerow([qid, doc, rank, score, focus])

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
