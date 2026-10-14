
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import load_embeddings
from evaluate_combinations import paired_difference
from evaluate_detection import (
    ATTACKS, N_BOOTSTRAPS, SEED, auroc, cross_validated_scores, tpr_at_fpr,
)
from generate_reverse_questions import EMBEDDING_NAME, N_QUESTIONS

# Detection with reverse questions (idea 1, added after the main results).
# The two signals were fixed before the results were seen:
# - question_focus: mean cosine similarity between the questions generated
#   from a passage and the user's question. A poisoned passage is written
#   for the user's question, so its questions should be close to it.
# - question_sameness: mean cosine similarity between the generated
#   questions themselves. A poisoned passage should answer only one
#   question, so its questions should be alike.
# A higher value is more suspicious for both. Because the analysis of
# relevance consistency showed that a signal can work only through the
# retrieval score, each signal is also tested inside bins of equal
# retrieval score and of equal passage length, and added to the baselines.
TRUST_DIR = Path("data_processed/trust")
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
ADAPTIVE_SCORES_PATH = TRUST_DIR / "adaptive_attack_scores.csv"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
ADAPTIVE_PATH = TRUST_DIR / "adaptive_attack_passages.jsonl"
OUTPUT_PATH = TRUST_DIR / "reverse_question_results.json"

SETTINGS = (*ATTACKS, "reworded")
SIGNALS = ("question_focus", "question_sameness")
# Baselines, and relevance consistency (z_drop) for comparison.
BASELINES = (
    "retrieval_score", "set_density", "shortness", "reader_support",
    "z_drop",
)
# Logistic regression combinations, cross-validated by question inside
# each setting.
COMBINATIONS = {
    "retrieval+focus": ["retrieval_score", "question_focus"],
    "retrieval+sameness": ["retrieval_score", "question_sameness"],
    "retrieval+both": ["retrieval_score", *SIGNALS],
    "set_density+both": ["set_density", *SIGNALS],
    "three_baselines": ["retrieval_score", "set_density", "reader_support"],
    "three_baselines+both": [
        "retrieval_score", "set_density", "reader_support", *SIGNALS,
    ],
}
COMPARISONS = [
    ("question_focus", "z_drop"),
    ("question_focus", "retrieval_score"),
    ("question_sameness", "retrieval_score"),
    ("retrieval+both", "retrieval_score"),
    ("set_density+both", "set_density"),
    ("three_baselines+both", "three_baselines"),
]
N_LEVEL_BINS = 10
MIN_PER_GROUP = 10


def normalise(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def within_bins(labels, scores, level):
    """Mean AUROC inside bins of equal size on `level` (no labels used to
    make the bins); bins with too few passages of either class are left
    out."""
    edges = np.quantile(level, np.linspace(0, 1, N_LEVEL_BINS + 1))
    bins = np.clip(np.digitize(level, edges[1:-1]), 0, N_LEVEL_BINS - 1)
    values = []

    for b in range(N_LEVEL_BINS):
        in_bin = bins == b
        n_poisoned = labels[in_bin].sum()

        if min(n_poisoned, in_bin.sum() - n_poisoned) >= MIN_PER_GROUP:
            values.append(auroc(labels[in_bin], scores[in_bin]))

    return {"auroc": float(np.mean(values)) if values else None,
            "bins_used": len(values)}


def main():
    question_emb, question_index = load_embeddings(EMBEDDING_NAME)
    query_emb, query_index = load_embeddings("query")

    def signals(pid, qid):
        generated = normalise(np.stack([
            question_emb[question_index[f"{pid}#{i}"]]
            for i in range(N_QUESTIONS)
        ]))
        user = normalise(query_emb[query_index[f"{qid}_original"]])
        pairs = generated @ generated.T
        upper = pairs[np.triu_indices(N_QUESTIONS, k=1)]
        return float((generated @ user).mean()), float(upper.mean())

    texts = {}

    for path in (EVAL_PATH, ADAPTIVE_PATH):
        with path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                texts[p["passage_id"]] = p["text"]

    with SCORES_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with ADAPTIVE_SCORES_PATH.open(encoding="utf-8") as f:
        adaptive = list(csv.DictReader(f))

    def features(table, density_column):
        values = {
            "retrieval_score": np.array([float(r["score_original"])
                                         for r in table]),
            "set_density": np.array([float(r[density_column])
                                     for r in table]),
            "shortness": -np.array([len(texts[r["passage_id"]].split())
                                    for r in table], dtype=float),
            "reader_support": np.array([float(r["support_original"])
                                        for r in table]),
            "z_drop": np.array([float(r["z_drop"]) for r in table]),
        }
        pairs = np.array([signals(r["passage_id"], r["query_id"])
                          for r in table])
        values["question_focus"] = pairs[:, 0]
        values["question_sameness"] = pairs[:, 1]
        return values

    attacks = np.array([r["attack"] for r in rows])
    settings = {}

    for attack in ATTACKS:
        keep = (attacks == "none") | (attacks == attack)
        table = [r for r, k in zip(rows, keep) if k]
        settings[attack] = (table, features(table, f"set_density_{attack}"))

    settings["reworded"] = (adaptive, features(adaptive, "set_density"))

    rng = np.random.default_rng(SEED)
    results = {}

    for setting, (table, values) in settings.items():
        labels = np.array([int(r["label"]) for r in table])
        groups = np.array([r["query_id"] for r in table])
        scores = dict(values)

        for name, columns in COMBINATIONS.items():
            scores[name] = cross_validated_scores(
                np.stack([values[c] for c in columns], axis=1),
                labels, groups,
            )

        questions = np.unique(groups)
        rows_of = {q: np.flatnonzero(groups == q) for q in questions}
        samples = [
            np.concatenate([rows_of[q]
                            for q in rng.choice(questions, len(questions))])
            for _ in range(N_BOOTSTRAPS)
        ]

        def auroc_ci(s):
            low, high = np.percentile(
                [auroc(labels[i], s[i]) for i in samples], [2.5, 97.5]
            )
            return [float(low), float(high)]

        length = -values["shortness"]
        results[setting] = {
            "passages": len(table),
            "poisoned": int(labels.sum()),
            "methods": {
                name: {
                    "auroc": auroc(labels, s),
                    "auroc_ci_95": auroc_ci(s),
                    "tpr_at_5pct_fpr": tpr_at_fpr(labels, s),
                }
                for name, s in scores.items()
            },
            "controls": {
                name: {
                    "same_retrieval_score": within_bins(
                        labels, values[name], values["retrieval_score"]
                    ),
                    "same_length": within_bins(labels, values[name], length),
                }
                for name in SIGNALS
            },
            "mean": {
                name: {
                    "benign": float(values[name][labels == 0].mean()),
                    "poisoned": float(values[name][labels == 1].mean()),
                }
                for name in SIGNALS
            },
            "comparisons": {
                f"{a} - {b}": paired_difference(
                    labels, groups, scores[a], scores[b], rng
                )
                for a, b in COMPARISONS
            },
        }

    results["setup"] = {
        "signals_fixed_before_results": list(SIGNALS),
        "questions_per_passage": N_QUESTIONS,
        "level_bins": N_LEVEL_BINS,
        "min_per_group": MIN_PER_GROUP,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nREVERSE QUESTIONS (AUROC [95% CI], TPR at 5% FPR)")
    print("=" * 78)

    for setting in SETTINGS:
        r = results[setting]
        print(f"\n{setting} ({r['poisoned']} poisoned, "
              f"{r['passages'] - r['poisoned']} benign)")

        for name, m in r["methods"].items():
            print(f"  {name:<24}{m['auroc']:.3f} "
                  f"[{m['auroc_ci_95'][0]:.3f}, {m['auroc_ci_95'][1]:.3f}]"
                  f"{m['tpr_at_5pct_fpr']:>8.3f}")

        for name in SIGNALS:
            c = r["controls"][name]
            mean = r["mean"][name]
            print(f"  {name}: benign {mean['benign']:.3f}, poisoned "
                  f"{mean['poisoned']:.3f}; same retrieval score "
                  f"{c['same_retrieval_score']['auroc']:.3f} "
                  f"({c['same_retrieval_score']['bins_used']} bins), "
                  f"same length {c['same_length']['auroc']:.3f} "
                  f"({c['same_length']['bins_used']} bins)")

        for name, c in r["comparisons"].items():
            print(f"  {name:<44}{c['auroc_difference']:+.3f} "
                  f"[{c['ci_95'][0]:+.3f}, {c['ci_95'][1]:+.3f}] "
                  f"p={c['p_value']:.3f}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
