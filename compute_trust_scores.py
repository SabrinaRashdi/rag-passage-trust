
import csv
import json
from pathlib import Path

import numpy as np

from src.trust import answer_f1, consistency_drop, pool_rank, z_score

DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
ANSWER_SUPPORT_PATH = TRUST_DIR / "answer_support.csv"
PERPLEXITY_PATH = TRUST_DIR / "perplexity.csv"
DATASET_PATH = DATA_DIR / "nq_passage_dataset.json"
OUTPUT_PATH = TRUST_DIR / "trust_scores.csv"

VARIANTS = ("original", "para1", "para2", "para3")
ATTACKS = ("prefix", "noprefix")
# Set-level baseline: number of nearest neighbours in the retrieved set.
SET_NEIGHBOURS = 3


def load_embeddings(name):
    embeddings = np.load(TRUST_DIR / f"{name}_embeddings.npy")

    with (TRUST_DIR / f"{name}_embedding_ids.json").open(
        encoding="utf-8"
    ) as f:
        ids = json.load(f)

    assert len(ids) == len(embeddings)
    return embeddings, {pid: i for i, pid in enumerate(ids)}


def main():
    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")
    query_emb, query_index = load_embeddings("query")

    with EVAL_PATH.open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    # Reader outputs from generate_answer_support.py
    with ANSWER_SUPPORT_PATH.open(encoding="utf-8") as f:
        reader = {row["passage_id"]: row for row in csv.DictReader(f)}

    # GPT-2 perplexity from generate_perplexity.py
    with PERPLEXITY_PATH.open(encoding="utf-8") as f:
        perplexity = {
            row["passage_id"]: float(row["log_perplexity"])
            for row in csv.DictReader(f)
        }

    by_query = {}

    for passage in passages:
        by_query.setdefault(passage["query_id"], []).append(passage)

    rows = []

    for qid, group in by_query.items():
        # Dot-product scores of every query variant against the clean pool.
        variants = np.stack([
            query_emb[query_index[f"{qid}_{v}"]] for v in VARIANTS
        ])
        clean_scores = variants @ pool_emb.T

        group_rows, group_embeddings = [], []

        for passage in group:
            if passage["label"] == 1:
                embedding = adv_emb[adv_index[passage["passage_id"]]]
                own_index = None
            else:
                own_index = pool_index[passage["doc_id"]]
                embedding = pool_emb[own_index]

            scores = variants @ embedding

            row = {
                "passage_id": passage["passage_id"],
                "query_id": qid,
                "label": passage["label"],
                "attack": passage["attack"],
                "is_gold": int(passage["is_gold"]),
                "retrieved_top10": int(passage["retrieved_top10"]),
            }

            for v, name in enumerate(VARIANTS):
                row[f"score_{name}"] = float(scores[v])
                row[f"z_{name}"] = z_score(scores[v], clean_scores[v])
                row[f"rank_{name}"] = pool_rank(
                    scores[v], clean_scores[v].copy(), own_index
                )

            # Answer support: does the passage give an answer, and is it
            # the same answer for every wording of the question?
            answers = reader[passage["passage_id"]]

            for name in VARIANTS:
                row[f"support_{name}"] = float(answers[f"support_{name}"])

            for name in VARIANTS[1:]:
                row[f"answer_f1_{name}"] = answer_f1(
                    answers["answer_original"], answers[f"answer_{name}"]
                )

            row["log_perplexity"] = perplexity[passage["passage_id"]]

            group_rows.append(row)
            group_embeddings.append(embedding)

        # Set-level baseline (in the spirit of TrustRAG's clustering step):
        # how close a passage is to its nearest neighbours in the candidate
        # set. Poisoned passages for one question tend to form a dense group.
        group_embeddings = np.stack(group_embeddings)
        group_embeddings = group_embeddings / np.linalg.norm(
            group_embeddings, axis=1, keepdims=True
        )

        for attack in ATTACKS:
            members = [
                i for i, r in enumerate(group_rows)
                if r["attack"] in ("none", attack)
            ]
            similarity = (
                group_embeddings[members] @ group_embeddings[members].T
            )
            np.fill_diagonal(similarity, -np.inf)
            density = np.sort(similarity, axis=1)[:, -SET_NEIGHBOURS:].mean(
                axis=1
            )

            for r in group_rows:
                r[f"set_density_{attack}"] = float("nan")

            for i, value in zip(members, density):
                group_rows[i][f"set_density_{attack}"] = float(value)

        rows.extend(group_rows)

    # Relevance-consistency signals using all three paraphrases.
    paraphrases = VARIANTS[1:]

    def column(prefix, names):
        return np.array([[r[f"{prefix}_{n}"] for n in names] for r in rows])

    raw_drop = consistency_drop(
        column("score", ["original"])[:, 0], column("score", paraphrases)
    )
    z_drop = consistency_drop(
        column("z", ["original"])[:, 0], column("z", paraphrases)
    )
    # Ranks are compared on a log scale; a drop means a worse (higher) rank.
    rank_drop = -consistency_drop(
        np.log(column("rank", ["original"])[:, 0]),
        np.log(column("rank", paraphrases)),
    )

    # Answer-support consistency signals.
    support_drop = consistency_drop(
        column("support", ["original"])[:, 0], column("support", paraphrases)
    )
    answer_disagreement = 1 - column("answer_f1", paraphrases).mean(axis=1)

    for i, row in enumerate(rows):
        row["raw_drop"] = float(raw_drop[i])
        row["z_drop"] = float(z_drop[i])
        row["rank_drop"] = float(rank_drop[i])
        row["support_drop"] = float(support_drop[i])
        row["answer_disagreement"] = float(answer_disagreement[i])

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Check our embeddings against the scores saved by PoisonedRAG.
    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    saved = {
        f"{qid}::{p['doc_id']}": p["score"]
        for qid, record in dataset.items()
        for p in record["benign_passages"]
    }
    differences = np.array([
        abs(r["score_original"] - saved[r["passage_id"]])
        for r in rows if r["passage_id"] in saved
    ])

    print("\nTRUST SCORES")
    print("=" * 45)
    print("Questions:", len(by_query))
    print("Scored passages:", len(rows))
    print("Check against PoisonedRAG scores:")
    print("  passages compared:", len(differences))
    print("  max difference:", float(differences.max()))
    print("  mean difference:", float(differences.mean()))

    for name, values in (
        ("raw_drop", raw_drop), ("z_drop", z_drop), ("rank_drop", rank_drop),
        ("support_original", column("support", ["original"])[:, 0]),
        ("support_drop", support_drop),
        ("answer_disagreement", answer_disagreement),
    ):
        print(f"\nMean {name}:")
        for label, attack in (
            ("benign", "none"),
            ("poisoned (prefix)", "prefix"),
            ("poisoned (noprefix)", "noprefix"),
        ):
            mask = np.array([r["attack"] == attack for r in rows])
            print(f"  {label}: {values[mask].mean():.4f}")

        gold = np.array([r["is_gold"] == 1 for r in rows])
        print(f"  benign gold only: {values[gold].mean():.4f}")

    print("\nOutput file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
