
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Automatic meaning check for the paraphrases: a natural language inference
# (NLI) model tests whether the original question and the paraphrase entail
# each other. A paraphrase passes only if entailment holds in both directions.
MODEL_NAME = "FacebookAI/roberta-large-mnli"
DATA_DIR = Path("data_processed")

QUERIES_PATH = DATA_DIR / "experiment_queries.json"
OUTPUT_PATH = DATA_DIR / "paraphrase_nli_check.csv"
SUMMARY_PATH = DATA_DIR / "paraphrase_nli_summary.json"
# Manual review files; compared with the NLI result once they are filled.
REVIEW_PATHS = {
    "sanduni": DATA_DIR / "paraphrase_manual_review_sanduni.csv",
    "sabrina": DATA_DIR / "paraphrase_manual_review_sabrina.csv",
}

PARAPHRASES = ("para1", "para2", "para3")
BATCH_SIZE = 16


def entailment_probability(premises, hypotheses, tokenizer, model, device):
    entailment = model.config.label2id["ENTAILMENT"]
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(premises), BATCH_SIZE):
            inputs = tokenizer(
                premises[start:start + BATCH_SIZE],
                hypotheses[start:start + BATCH_SIZE],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits
            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, entailment].cpu().tolist()
            )

    return np.array(probabilities)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading NLI model on", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    query_ids = sorted({key.rsplit("_", 1)[0] for key in queries})

    items = [
        (f"{qid}_{name}", qid, queries[f"{qid}_original"],
         queries[f"{qid}_{name}"])
        for qid in query_ids
        for name in PARAPHRASES
    ]
    originals = [item[2] for item in items]
    paraphrases = [item[3] for item in items]

    forward = entailment_probability(
        originals, paraphrases, tokenizer, model, device
    )
    backward = entailment_probability(
        paraphrases, originals, tokenizer, model, device
    )
    # Entailment is the most likely of the three NLI classes when its
    # probability is above 0.5.
    passed = (forward > 0.5) & (backward > 0.5)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "query_id", "original_question", "paraphrase",
            "entail_original_to_paraphrase", "entail_paraphrase_to_original",
            "nli_same_meaning",
        ])

        for item, fwd, bwd, ok in zip(items, forward, backward, passed):
            writer.writerow([
                *item, round(float(fwd), 4), round(float(bwd), 4),
                "yes" if ok else "no",
            ])

    nli_label = {item[0]: bool(ok) for item, ok in zip(items, passed)}
    by_question = {}

    for item, ok in zip(items, passed):
        by_question.setdefault(item[1], []).append(ok)

    summary = {
        "model": MODEL_NAME,
        "paraphrases": len(items),
        "passed": int(passed.sum()),
        "pass_rate": float(passed.mean()),
        "failed_original_to_paraphrase": int((forward <= 0.5).sum()),
        "failed_paraphrase_to_original": int((backward <= 0.5).sum()),
        "questions_with_a_failed_paraphrase": sum(
            not all(v) for v in by_question.values()
        ),
        "agreement_with_manual_review": {},
    }

    for reviewer, path in REVIEW_PATHS.items():
        with path.open(encoding="utf-8-sig") as f:
            reviewed = [
                row for row in csv.DictReader(f)
                if row["same_meaning"].strip().lower() in ("yes", "no")
            ]

        if not reviewed:
            continue

        human = np.array([
            row["same_meaning"].strip().lower() == "yes" for row in reviewed
        ])
        machine = np.array([nli_label[row["item_id"]] for row in reviewed])

        summary["agreement_with_manual_review"][reviewer] = {
            "rows": len(reviewed),
            "human_yes_rate": float(human.mean()),
            "nli_yes_rate": float(machine.mean()),
            "agreement": float((human == machine).mean()),
            "cohen_kappa": float(cohen_kappa_score(human, machine)),
            "human_yes_nli_yes": int((human & machine).sum()),
            "human_yes_nli_no": int((human & ~machine).sum()),
            "human_no_nli_yes": int((~human & machine).sum()),
            "human_no_nli_no": int((~human & ~machine).sum()),
        }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPARAPHRASE NLI CHECK")
    print("=" * 55)
    print("Paraphrases checked:", summary["paraphrases"])
    print(f"Passed (entailment both ways): {summary['passed']} "
          f"({summary['pass_rate']:.1%})")
    print("Failed original -> paraphrase:",
          summary["failed_original_to_paraphrase"])
    print("Failed paraphrase -> original:",
          summary["failed_paraphrase_to_original"])
    print("Questions with a failed paraphrase:",
          summary["questions_with_a_failed_paraphrase"])

    for reviewer, a in summary["agreement_with_manual_review"].items():
        print(f"\nCompared with manual review ({reviewer}, {a['rows']} rows)")
        print(f"  Human yes rate: {a['human_yes_rate']:.1%}, "
              f"NLI yes rate: {a['nli_yes_rate']:.1%}")
        print(f"  Agreement: {a['agreement']:.1%}, "
              f"Cohen's kappa: {a['cohen_kappa']:.3f}")
        print(f"  human yes / NLI yes: {a['human_yes_nli_yes']}, "
              f"human yes / NLI no: {a['human_yes_nli_no']}")
        print(f"  human no / NLI yes:  {a['human_no_nli_yes']}, "
              f"human no / NLI no:  {a['human_no_nli_no']}")

    print("\nOutput file:", OUTPUT_PATH)
    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
