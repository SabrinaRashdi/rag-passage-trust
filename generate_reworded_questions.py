
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

INPUT_PATH = "data_processed/nq_passage_dataset.json"
OUTPUT_PATH = "data_processed/nq_reworded_questions.json"

print("Loading Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions.")

def generate_paraphrases(question, num_versions=3):
    inputs = tokenizer(
        "paraphrase: " + question,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=8,
            num_return_sequences=6,
            do_sample=False,
            early_stopping=True
        )

    candidates = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    unique = []
    seen = {question.strip().casefold()}

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate.casefold() not in seen:
            unique.append(candidate)
            seen.add(candidate.casefold())

        if len(unique) == num_versions:
            break

    return unique


results = {}

for index, (qid, record) in enumerate(dataset.items(), 1):
    question = record["question"]

    print(f"[{index}/{len(dataset)}] Processing {qid}")

    reworded = generate_paraphrases(question)

    results[qid] = {
        "original_question": question,
        "correct_answer": record["correct_answer"],
        "incorrect_answer": record["incorrect_answer"],
        "reworded_questions": reworded
    }

    # Save progress after each question
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("\nGeneration completed!")
print("Questions processed:", len(results))
print("Output saved to:", OUTPUT_PATH)
