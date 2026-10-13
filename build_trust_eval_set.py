
import json
from pathlib import Path

DATA_DIR = Path("data_processed")
OUTPUT_DIR = DATA_DIR / "trust"

DATASET_PATH = DATA_DIR / "nq_passage_dataset.json"
POOL_PATH = DATA_DIR / "candidate_passages.jsonl"
QRELS_PATH = Path("datasets/nq/qrels/test.tsv")

OUTPUT_PATH = OUTPUT_DIR / "eval_passages.jsonl"


def passage_text(title, text):
    # Same passage format as the Contriever BEIR encoder used by PoisonedRAG.
    return title + " " + text if len(title) > 0 else text


def main():
    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    pool = {}

    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pool[str(record["_id"])] = record

    gold = {}

    with QRELS_PATH.open(encoding="utf-8") as f:
        next(f)  # Skip the header row

        for line in f:
            qid, docid, score = line.strip().split("\t")

            if int(score) > 0:
                gold.setdefault(qid, set()).add(docid)

    rows = []

    for qid, record in dataset.items():
        question = record["question"]
        gold_ids = gold.get(qid, set())
        benign_ids = [p["doc_id"] for p in record["benign_passages"]]

        # Benign passages: the top retrieved clean passages, plus any gold
        # passage that was not retrieved, so that every question has at
        # least one benign passage that really answers it.
        for doc_id in benign_ids + sorted(gold_ids - set(benign_ids)):
            doc = pool[doc_id]

            rows.append({
                "passage_id": f"{qid}::{doc_id}",
                "query_id": qid,
                "doc_id": doc_id,
                "label": 0,
                "is_gold": doc_id in gold_ids,
                "retrieved_top10": doc_id in benign_ids,
                "attack": "none",
                "text": passage_text(doc.get("title", ""), doc["text"]),
            })

        # Poisoned passages in two versions. "prefix" is the PoisonedRAG
        # black-box attack (src/attack.py): the target question is put in
        # front of the generated text. "noprefix" is the generated text only.
        for j, adv_text in enumerate(record["adversarial_passages"], 1):
            for attack, text in (
                ("prefix", question + "." + adv_text),
                ("noprefix", adv_text),
            ):
                rows.append({
                    "passage_id": f"{qid}::adv{j}_{attack}",
                    "query_id": qid,
                    "doc_id": f"adv{j}",
                    "label": 1,
                    "is_gold": False,
                    "retrieved_top10": False,
                    "attack": attack,
                    "text": text,
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    benign = [r for r in rows if r["label"] == 0]

    print("\nTRUST EVALUATION SET")
    print("=" * 45)
    print("Questions:", len(dataset))
    print("Benign passages:", len(benign))
    print("  of which gold:", sum(r["is_gold"] for r in benign))
    print("  gold added (not in top 10):",
          sum(not r["retrieved_top10"] for r in benign))
    print("Poisoned passages (prefix):",
          sum(r["attack"] == "prefix" for r in rows))
    print("Poisoned passages (noprefix):",
          sum(r["attack"] == "noprefix" for r in rows))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
