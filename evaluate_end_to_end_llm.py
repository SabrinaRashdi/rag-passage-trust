
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_end_to_end import (
    TRUST_DIR, defended_read_lists, print_results, score_answers,
)
from src.prompts import wrap_prompt

# The end-to-end test of evaluate_end_to_end.py with a generative LLM in
# place of the extractive reader. The retrieved lists, thresholds and
# removed passages are the same; the 5 passages read are given to the LLM
# with the PoisonedRAG prompt, and its answer is scored in the same way.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 32
# Answers are saved after each new passage list, so a stopped run can be
# continued and the results can be recomputed without the LLM.
ANSWERS_PATH = TRUST_DIR / "llm_answers.csv"
OUTPUT_PATH = TRUST_DIR / "end_to_end_llm_results.json"
FIELDS = ("query_id", "passage_ids", "answer")


def load_answers():
    if not ANSWERS_PATH.exists():
        return {}

    with ANSWERS_PATH.open(encoding="utf-8") as f:
        return {
            (row["query_id"], row["passage_ids"]): row["answer"]
            for row in csv.DictReader(f)
        }


def llm_answers(dataset, passages, runs):
    """The LLM answer for every passage list read in runs; answers saved
    earlier are reused. Returns answer_of(qid, read) for score_answers."""
    # Every distinct (question, passages read) pair; many defences read
    # the same passages.
    needed = {
        (qid, tuple(read))
        for methods in runs.values()
        for run in methods.values()
        for condition in ("clean", "attacked")
        for qid, (_, read) in run[condition].items()
    }
    answers = load_answers()
    missing = sorted(
        (qid, read) for qid, read in needed
        if (qid, " ".join(read)) not in answers
    )
    print(f"Passage lists: {len(needed)}, not yet answered: {len(missing)}")

    if missing:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16
        ).to(device).eval()

        new_file = not ANSWERS_PATH.exists()

        with ANSWERS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)

            if new_file:
                writer.writeheader()

            for i, (qid, read) in enumerate(missing, 1):
                prompt = wrap_prompt(
                    dataset[qid]["question"],
                    [passages[pid]["text"] for pid in read],
                    prompt_id=4,
                )
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt",
                    return_dict=True,
                ).to(device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )

                answer = tokenizer.decode(
                    output[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                writer.writerow({"query_id": qid,
                                 "passage_ids": " ".join(read),
                                 "answer": answer})
                f.flush()
                answers[qid, " ".join(read)] = answer

                if i % 50 == 0 or i == len(missing):
                    print(f"  answered {i}/{len(missing)}")

    def answer_of(qid, read):
        return answers[qid, " ".join(read)]

    return answer_of


def main():
    dataset, passages, questions, runs, setup = defended_read_lists()
    answer_of = llm_answers(dataset, passages, runs)
    results = score_answers(dataset, passages, questions, runs, answer_of)
    results["setup"] = {**setup, "reader": MODEL_NAME,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "decoding": "greedy"}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_results(results, f"END-TO-END EFFECT OF EACH DEFENCE ({MODEL_NAME})")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
