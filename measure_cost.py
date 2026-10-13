
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, AutoTokenizer,
)

from compute_trust_scores import SET_NEIGHBOURS, VARIANTS, load_embeddings
from generate_answer_support import MODEL_NAME as READER_NAME
from generate_answer_support import read
from generate_perplexity import MAX_LENGTH as PERPLEXITY_MAX_LENGTH
from generate_perplexity import MODEL_NAME as PERPLEXITY_NAME
from generate_trust_embeddings import MODEL_NAME as RETRIEVER_NAME
from generate_trust_embeddings import encode
from src.contriever_src.contriever import Contriever
from src.trust import reference_stats, set_density

# Cost of each detection method: extra time per question at query time.
# Passage embeddings are taken from the index (computed offline), as in a
# normal dense retrieval system, so they are not part of the cost.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
OUTPUT_PATH = TRUST_DIR / "cost_results.json"

# Same model and generation settings as generate_reworded_questions.py.
PARAPHRASER_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
N_PARAPHRASES = len(VARIANTS) - 1
# The candidate set of a question: its benign passages plus the poisoned
# passages of the real attack.
ATTACK = "prefix"

# Which timed stages each method needs on top of plain retrieval.
METHOD_STAGES = {
    "retrieval_z": [],
    "shortness": [],
    "set_density": ["set_density"],
    "perplexity": ["perplexity"],
    "reader_support": ["reader_original"],
    "z_drop_k3": [
        "generate_paraphrases", "encode_paraphrases", "search_paraphrases",
    ],
    "combined_consistency": [
        "generate_paraphrases", "encode_paraphrases", "search_paraphrases",
        "reader_original", "reader_paraphrases",
    ],
}
PLAIN_RETRIEVAL = ["encode_original", "search_original"]


def timed(function, questions):
    """Seconds taken by function(question) for every question. The first
    question is run once before timing so model start-up is not counted."""
    function(questions[0])
    seconds = []

    for question in questions:
        start = time.perf_counter()
        function(question)
        seconds.append(time.perf_counter() - start)

    return np.array(seconds)


def release(*objects):
    del objects
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Measuring on", device)

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")

    candidates = {}

    for passage in passages:
        if passage["attack"] in ("none", ATTACK):
            candidates.setdefault(passage["query_id"], []).append(passage)

    questions = sorted(candidates)
    paraphrase_names = VARIANTS[1:]
    stages = {}

    # 1. Retriever: encode the question wordings and search the pool.
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_NAME)
    model = Contriever.from_pretrained(RETRIEVER_NAME).to(device)
    model.eval()

    def encode_variants(qid, names):
        return encode(
            [queries[f"{qid}_{n}"] for n in names],
            tokenizer, model, device, "query",
        )

    def search(vectors):
        # Scores against the whole pool and the top-100 statistics.
        for scores in vectors @ pool_emb.T:
            reference_stats(scores)

    stages["encode_original"] = timed(
        lambda q: encode_variants(q, VARIANTS[:1]), questions
    )
    stages["encode_paraphrases"] = timed(
        lambda q: encode_variants(q, paraphrase_names), questions
    )

    vectors = {q: encode_variants(q, VARIANTS) for q in questions}
    stages["search_original"] = timed(
        lambda q: search(vectors[q][:1]), questions
    )
    stages["search_paraphrases"] = timed(
        lambda q: search(vectors[q][1:]), questions
    )
    release(tokenizer, model)

    # 2. Paraphrase generation.
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_NAME)
    model = model.to(device)
    model.eval()

    def generate_paraphrases(qid):
        inputs = tokenizer(
            "paraphrase: " + queries[f"{qid}_original"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=8,
                num_return_sequences=6,
                do_sample=False,
                early_stopping=True,
            )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    stages["generate_paraphrases"] = timed(generate_paraphrases, questions)
    release(tokenizer, model)

    # 3. Reader: every candidate passage with one or three wordings.
    tokenizer = AutoTokenizer.from_pretrained(READER_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(READER_NAME)
    model = model.to(device)
    model.eval()

    def run_reader(qid, names):
        texts = [p["text"] for p in candidates[qid]]

        for name in names:
            read(
                [queries[f"{qid}_{name}"]] * len(texts), texts,
                tokenizer, model, device,
            )

    stages["reader_original"] = timed(
        lambda q: run_reader(q, VARIANTS[:1]), questions
    )
    stages["reader_paraphrases"] = timed(
        lambda q: run_reader(q, paraphrase_names), questions
    )
    release(tokenizer, model)

    # 4. Perplexity of every candidate passage.
    tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_NAME)
    model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_NAME).to(device)
    model.eval()

    def run_perplexity(qid):
        with torch.inference_mode():
            for passage in candidates[qid]:
                inputs = tokenizer(
                    passage["text"],
                    truncation=True,
                    max_length=PERPLEXITY_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)
                float(model(**inputs, labels=inputs["input_ids"]).loss)

    stages["perplexity"] = timed(run_perplexity, questions)
    release(tokenizer, model)

    # 5. Set density: needs only the stored passage embeddings.
    def run_set_density(qid):
        set_density(
            np.stack([
                adv_emb[adv_index[p["passage_id"]]] if p["label"] == 1
                else pool_emb[pool_index[p["doc_id"]]]
                for p in candidates[qid]
            ]),
            SET_NEIGHBOURS,
        )

    stages["set_density"] = timed(run_set_density, questions)

    def summarise(seconds):
        return {
            "median_ms": float(np.median(seconds) * 1000),
            "mean_ms": float(np.mean(seconds) * 1000),
            "p95_ms": float(np.percentile(seconds, 95) * 1000),
        }

    plain = sum(stages[name] for name in PLAIN_RETRIEVAL)
    results = {
        "device": device,
        "machine": platform.platform(),
        "questions": len(questions),
        "pool_size": len(pool_emb),
        "candidates_per_question": float(
            np.mean([len(candidates[q]) for q in questions])
        ),
        "stages": {name: summarise(s) for name, s in stages.items()},
        "plain_retrieval": summarise(plain),
        "extra_cost": {},
    }

    for method, names in METHOD_STAGES.items():
        extra = sum((stages[name] for name in names), np.zeros(len(plain)))
        results["extra_cost"][method] = {
            **summarise(extra),
            "times_plain_retrieval": float(
                np.median(extra) / np.median(plain)
            ),
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nCOST PER QUESTION")
    print("=" * 66)
    print(f"Device: {device}, questions: {len(questions)}, "
          f"pool: {len(pool_emb)}, candidates per question: "
          f"{results['candidates_per_question']:.1f}")

    print(f"\n{'stage':<26}{'median ms':>12}{'mean ms':>12}{'p95 ms':>12}")

    for name, s in results["stages"].items():
        print(f"{name:<26}{s['median_ms']:>12.2f}{s['mean_ms']:>12.2f}"
              f"{s['p95_ms']:>12.2f}")

    print(f"\nPlain retrieval (no defence): "
          f"{results['plain_retrieval']['median_ms']:.2f} ms")
    print(f"\n{'method':<26}{'extra ms':>12}{'x plain retrieval':>20}")

    for method, s in results["extra_cost"].items():
        print(f"{method:<26}{s['median_ms']:>12.2f}"
              f"{s['times_plain_retrieval']:>20.1f}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
