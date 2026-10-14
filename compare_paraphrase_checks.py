
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModel, AutoTokenizer

from compute_trust_scores import load_embeddings
from evaluate_detection import N_BOOTSTRAPS, SEED, auroc

# How well do the automatic paraphrase checks agree with the human review?
# Human label: "good" if no reviewer said the meaning changed
# (paraphrase_review_combined.csv). Automatic checks:
# - NLI: entailment both ways (check_paraphrases_nli.py); score = the
#   smaller of the two entailment probabilities, rule = both above 0.5.
# - MiniLM: cosine similarity, as in validate_paraphrases.py; rule = 0.75.
# - Contriever: cosine similarity of the query embeddings the retriever
#   uses; it has no rule threshold, so only its AUROC is reported.
DATA_DIR = Path("data_processed")
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
LABELS_PATH = DATA_DIR / "paraphrase_review_combined.csv"
NLI_PATH = DATA_DIR / "paraphrase_nli_check.csv"
OUTPUT_PATH = DATA_DIR / "paraphrase_check_comparison.json"

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_THRESHOLD = 0.75
NLI_THRESHOLD = 0.5


def minilm_similarity(originals, paraphrases):
    """Cosine similarity with mean pooling, as sentence-transformers does
    for all-MiniLM-L6-v2."""
    tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)
    model = AutoModel.from_pretrained(MINILM_NAME)
    model.eval()

    def encode(texts):
        inputs = tokenizer(texts, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")

        with torch.inference_mode():
            tokens = model(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (tokens * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, dim=-1).numpy()

    return (encode(originals) * encode(paraphrases)).sum(axis=1)


def auroc_with_ci(labels, scores, groups, rng):
    """AUROC of a score for predicting the human "good" label, with a
    bootstrap over questions (paraphrases of one question are not
    independent)."""
    questions = np.unique(groups)
    rows_of = {q: np.flatnonzero(groups == q) for q in questions}
    values = []

    for _ in range(N_BOOTSTRAPS):
        sample = np.concatenate([
            rows_of[q] for q in rng.choice(questions, size=len(questions))
        ])
        values.append(auroc(labels[sample], scores[sample]))

    low, high = np.percentile(values, [2.5, 97.5])
    return {"auroc": auroc(labels, scores), "ci_95": [float(low), float(high)]}


def rule_agreement(labels, passed):
    good, ok = labels == 1, passed
    return {
        "pass_rate": float(ok.mean()),
        "agreement": float((good == ok).mean()),
        "cohen_kappa": float(cohen_kappa_score(good, ok)),
        "human_good_check_pass": int((good & ok).sum()),
        "human_good_check_fail": int((good & ~ok).sum()),
        "human_bad_check_pass": int((~good & ok).sum()),
        "human_bad_check_fail": int((~good & ~ok).sum()),
    }


def main():
    with LABELS_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with NLI_PATH.open(encoding="utf-8") as f:
        nli = {row["item_id"]: row for row in csv.DictReader(f)}

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    # The review file must hold the paraphrases used in the experiment.
    for row in rows:
        assert row["paraphrase"] == queries[row["item_id"]], row["item_id"]

    items = [row["item_id"] for row in rows]
    groups = np.array([row["query_id"] for row in rows])
    labels = np.array([row["same_meaning_combined"] == "yes" for row in rows],
                      dtype=int)

    originals = [queries[f"{q}_original"] for q in groups]
    paraphrases = [queries[i] for i in items]

    query_emb, query_index = load_embeddings("query")
    first = query_emb[[query_index[f"{q}_original"] for q in groups]]
    second = query_emb[[query_index[i] for i in items]]
    contriever = (first * second).sum(1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )

    scores = {
        "nli": np.array([
            min(float(nli[i]["entail_original_to_paraphrase"]),
                float(nli[i]["entail_paraphrase_to_original"]))
            for i in items
        ]),
        "minilm": minilm_similarity(originals, paraphrases),
        "contriever": contriever,
    }
    rules = {
        "nli": scores["nli"] > NLI_THRESHOLD,
        "minilm": scores["minilm"] >= MINILM_THRESHOLD,
    }

    rng = np.random.default_rng(SEED)
    results = {
        "paraphrases": len(rows),
        "questions": len(np.unique(groups)),
        "human_good": int(labels.sum()),
        "thresholds": {"nli": NLI_THRESHOLD, "minilm": MINILM_THRESHOLD},
        "auroc": {
            name: auroc_with_ci(labels, s, groups, rng)
            for name, s in scores.items()
        },
        "mean_score": {
            name: {"human_good": float(s[labels == 1].mean()),
                   "human_bad": float(s[labels == 0].mean())}
            for name, s in scores.items()
        },
        "rule": {
            name: rule_agreement(labels, passed)
            for name, passed in rules.items()
        },
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAUTOMATIC PARAPHRASE CHECKS AGAINST THE HUMAN REVIEW")
    print(f"({results['paraphrases']} paraphrases, {results['questions']} "
          f"questions, {results['human_good']} judged good by humans)")
    print("=" * 70)
    print(f"{'check':<12}{'AUROC':>8}{'95% CI':>18}"
          f"{'mean good':>12}{'mean bad':>11}")

    for name, a in results["auroc"].items():
        m = results["mean_score"][name]
        print(f"{name:<12}{a['auroc']:>8.3f}"
              f"   [{a['ci_95'][0]:.3f}, {a['ci_95'][1]:.3f}]"
              f"{m['human_good']:>12.3f}{m['human_bad']:>11.3f}")

    print(f"\n{'rule':<12}{'pass':>7}{'agree':>8}{'kappa':>8}"
          f"{'good/pass':>11}{'good/fail':>11}{'bad/pass':>10}{'bad/fail':>10}")

    for name, r in results["rule"].items():
        print(f"{name:<12}{r['pass_rate']:>7.0%}{r['agreement']:>8.0%}"
              f"{r['cohen_kappa']:>8.3f}{r['human_good_check_pass']:>11d}"
              f"{r['human_good_check_fail']:>11d}"
              f"{r['human_bad_check_pass']:>10d}"
              f"{r['human_bad_check_fail']:>10d}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
