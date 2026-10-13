
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import SET_NEIGHBOURS, load_embeddings
from evaluate_detection import ATTACKS, auroc, cross_validated_scores
from src.trust import set_density

# How does detection change when the attacker injects fewer poisoned
# passages per question? PoisonedRAG injects 5; here we keep 1 to 5 of them.
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
OUTPUT_PATH = TRUST_DIR / "poison_count_results.json"

POISON_COUNTS = (1, 2, 3, 4, 5)
# Which poisoned passages are kept is random, so each count is repeated.
N_DRAWS = 20
SEED = 42

# Per-passage methods: column in trust_scores.csv and its sign
# (-1 when a smaller value is the more suspicious one).
PASSAGE_METHODS = {
    "retrieval_z": ("z_original", 1),
    "perplexity": ("log_perplexity", 1),
    "shortness": ("num_words", -1),
    "reader_support": ("support_original", 1),
    "z_drop_k3": ("z_drop", 1),
}
COMBINED_FEATURES = ("z_drop", "support_drop", "answer_disagreement")


def main():
    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")

    with EVAL_PATH.open(encoding="utf-8") as f:
        doc_ids = {
            p["passage_id"]: p.get("doc_id")
            for p in (json.loads(line) for line in f)
        }

    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = np.array([int(r["label"]) for r in rows])
    attacks = np.array([r["attack"] for r in rows])
    query_ids = np.array([r["query_id"] for r in rows])

    embeddings = np.stack([
        adv_emb[adv_index[r["passage_id"]]] if r["label"] == "1"
        else pool_emb[pool_index[doc_ids[r["passage_id"]]]]
        for r in rows
    ])

    columns = {
        name: np.array([float(r[name]) for r in rows])
        for name in (
            {column for column, _ in PASSAGE_METHODS.values()}
            | set(COMBINED_FEATURES)
        )
    }
    combined_features = np.stack(
        [columns[name] for name in COMBINED_FEATURES], axis=1
    )

    rng = np.random.default_rng(SEED)
    results = {}

    for attack in ATTACKS:
        results[attack] = {}

        for count in POISON_COUNTS:
            draws = {
                name: [] for name in
                (*PASSAGE_METHODS, "combined_consistency", "set_density")
            }

            for _ in range(N_DRAWS):
                keep = attacks == "none"

                for qid in np.unique(query_ids):
                    poisoned = np.flatnonzero(
                        (query_ids == qid) & (attacks == attack)
                    )
                    keep[rng.choice(poisoned, size=count, replace=False)] = True

                kept = np.flatnonzero(keep)
                y = labels[kept]

                for name, (column, sign) in PASSAGE_METHODS.items():
                    draws[name].append(
                        auroc(y, sign * columns[column][kept])
                    )

                draws["combined_consistency"].append(auroc(
                    y,
                    cross_validated_scores(
                        combined_features[kept], y, query_ids[kept]
                    ),
                ))

                # Density is recomputed inside each question's smaller set.
                density = np.zeros(len(kept))

                for qid in np.unique(query_ids[kept]):
                    members = np.flatnonzero(query_ids[kept] == qid)
                    density[members] = set_density(
                        embeddings[kept[members]], SET_NEIGHBOURS
                    )

                draws["set_density"].append(auroc(y, density))

            results[attack][str(count)] = {
                name: {
                    "auroc_mean": float(np.mean(values)),
                    "auroc_std": float(np.std(values)),
                }
                for name, values in draws.items()
            }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAUROC BY NUMBER OF POISONED PASSAGES PER QUESTION")
    print(f"(all benign negatives, mean over {N_DRAWS} random draws)")
    print("=" * 70)

    for attack, by_count in results.items():
        print(f"\nAttack: {attack}")
        print(f"{'method':<24}"
              + "".join(f"{'n=' + c:>9}" for c in by_count))

        for name in by_count["1"]:
            print(f"{name:<24}" + "".join(
                f"{by_count[c][name]['auroc_mean']:>9.3f}" for c in by_count
            ))

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
