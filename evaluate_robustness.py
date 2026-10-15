
import csv
import json
from pathlib import Path

import numpy as np

from compute_trust_scores import SET_NEIGHBOURS, load_embeddings
from evaluate_adaptive_attack import trained_on_known_attack
from evaluate_detection import SEED, auroc, tpr_at_fpr
from src.trust import set_density

# Robustness: each method is scored on every attack setting we have, and
# its worst AUROC is reported. A defender does not know which attack will
# come, so a method is only as good as its worst setting. All settings use
# the same 1,085 benign passages as negatives. Trained methods are trained
# on Kind 1 only (the known attack), on other questions than they score.
#
# Two combinations without training were fixed before this script was
# run (post hoc with respect to the earlier results): each part is turned
# into the fraction of clean benign passages it is higher than, and the
# parts are averaged (rank_mean) or the largest is taken (rank_max).
# Added after seeing the first robustness results: the same two
# combinations without question focus, to check whether question focus is
# needed or the baselines alone are enough.
# Added after that (idea 3, list fixed before running): set-relative
# question focus, the focus minus the median focus of the same question's
# candidates, which removes how focused all passages of a question are;
# and rank_max/rank_mean with it in place of question focus.
TRUST_DIR = Path("data_processed/trust")
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
REVERSE_SCORES_PATH = TRUST_DIR / "reverse_question_scores.csv"
REWORDED_SCORES_PATH = TRUST_DIR / "adaptive_attack_scores.csv"
DIVERSE_SCORES_PATH = TRUST_DIR / "diverse_attack_scores.csv"
OUTPUT_PATH = TRUST_DIR / "robustness_results.json"

FEATURES = (
    "score_original", "num_words", "log_perplexity", "support_original",
    "z_drop", "support_drop", "answer_disagreement", "question_focus",
    "question_sameness", "set_density",
)
# Method: (feature, sign); sign -1 when a smaller value is more suspicious.
SINGLE_METHODS = {
    "retrieval_score": ("score_original", 1),
    "set_density": ("set_density", 1),
    "shortness": ("num_words", -1),
    "perplexity": ("log_perplexity", 1),
    "reader_support": ("support_original", 1),
    "z_drop": ("z_drop", 1),
    "question_focus": ("question_focus", 1),
    "question_sameness": ("question_sameness", 1),
}
TRAINED_METHODS = {
    "combined_consistency": ("z_drop", "support_drop", "answer_disagreement"),
    "retrieval+focus": ("score_original", "question_focus"),
}
RANK_PARTS = (
    "retrieval_score", "set_density", "reader_support", "question_focus",
)
RANK_COMBINATIONS = {
    "": RANK_PARTS,
    "_no_focus": tuple(p for p in RANK_PARTS if p != "question_focus"),
    "_relative": tuple(
        "question_focus_relative" if p == "question_focus" else p
        for p in RANK_PARTS
    ),
}
BASELINES = (
    "retrieval_score", "set_density", "shortness", "perplexity",
    "reader_support",
)
COMPARISONS = [
    (combination, other)
    for combination in ("rank_mean", "rank_max")
    for other in (*BASELINES, "question_focus", f"{combination}_no_focus")
] + [
    ("question_focus_relative", other) for other in BASELINES
] + [
    ("question_focus_relative", "question_focus"),
    ("rank_max_relative", "rank_max"),
    ("rank_mean_relative", "rank_mean"),
]
# Also compared inside every setting.
SETTING_COMPARISONS = [
    ("question_focus_relative", "question_focus"),
    ("rank_max_relative", "rank_max"),
]
# Settings with one poisoned passage per question: which one is kept is
# random, so the AUROC is the mean over several draws.
ONE_POISONED = {"kind1_one_poisoned": "prefix", "kind2_one_poisoned": "noprefix"}
N_DRAWS = 20
# More resamples than the other experiments: this family has many tests,
# and with 1,000 resamples the smallest possible p-value (2/1001) is too
# large to stay below 0.05 after the Holm correction.
N_BOOTSTRAPS = 10000


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(rows, extra=None):
    """Labels, question ids and feature arrays of a list of passages."""
    extra = extra or {}

    def value(row, name):
        if name in row and row[name] != "":
            return float(row[name])
        return float(extra[row["passage_id"]][name])

    return {
        "ids": [r["passage_id"] for r in rows],
        "labels": np.array([int(r["label"]) for r in rows]),
        "groups": np.array([r["query_id"] for r in rows]),
        **{name: np.array([value(r, name) for r in rows])
           for name in FEATURES},
    }


def subset(data, keep):
    return {
        key: ([data[key][i] for i in keep] if key == "ids"
              else data[key][keep])
        for key in data
    }


def densities(embeddings, groups):
    """Set density inside each question's candidate set."""
    density = np.zeros(len(groups))

    for qid in np.unique(groups):
        members = np.flatnonzero(groups == qid)
        density[members] = set_density(embeddings[members], SET_NEIGHBOURS)

    return density


def set_relative(values, groups):
    """Each value minus the median of its question's candidate set."""
    relative = np.zeros(len(values))

    for qid in np.unique(groups):
        members = groups == qid
        relative[members] = values[members] - np.median(values[members])

    return relative


def main():
    with EVAL_PATH.open(encoding="utf-8") as f:
        doc_ids = {
            p["passage_id"]: p.get("doc_id")
            for p in (json.loads(line) for line in f)
        }

    reverse = {r["passage_id"]: r for r in read_csv(REVERSE_SCORES_PATH)}
    rows = read_csv(SCORES_PATH)
    benign = [r for r in rows if r["attack"] == "none"]

    def with_density(attack):
        """Benign + this attack's poisoned passages, with the set density
        of the candidate sets under this attack."""
        return [
            {**r, "set_density": r[f"set_density_{attack}"]}
            for r in rows if r["attack"] in ("none", attack)
        ]

    settings = {
        "kind1": [table(with_density("prefix"), reverse)],
        "kind2": [table(with_density("noprefix"), reverse)],
    }

    pool_emb, pool_index = load_embeddings("pool")
    adv_emb, adv_index = load_embeddings("poisoned")

    def embedding(pid):
        if pid in adv_index:
            return adv_emb[adv_index[pid]]
        return pool_emb[pool_index[doc_ids[pid]]]

    rng = np.random.default_rng(SEED)

    for name, attack in ONE_POISONED.items():
        full = settings["kind1" if attack == "prefix" else "kind2"][0]
        embeddings = np.stack([embedding(pid) for pid in full["ids"]])
        settings[name] = []

        for _ in range(N_DRAWS):
            keep = full["labels"] == 0

            for qid in np.unique(full["groups"]):
                poisoned = np.flatnonzero(
                    (full["groups"] == qid) & (full["labels"] == 1)
                )
                keep[rng.choice(poisoned)] = True

            kept = np.flatnonzero(keep)
            draw = subset(full, kept)
            draw["set_density"] = densities(embeddings[kept], draw["groups"])
            settings[name].append(draw)

    settings["reworded"] = [table(read_csv(REWORDED_SCORES_PATH), reverse)]
    settings["diverse"] = [table(read_csv(DIVERSE_SCORES_PATH))]

    # Reference for the combinations without training: clean benign
    # passages, with set density computed among benign passages only.
    clean = subset(settings["kind1"][0],
                   np.flatnonzero(settings["kind1"][0]["labels"] == 0))
    clean["set_density"] = densities(
        np.stack([embedding(pid) for pid in clean["ids"]]), clean["groups"]
    )
    reference = {
        method: np.sort(sign * clean[feature])
        for method, (feature, sign) in SINGLE_METHODS.items()
    }
    reference["question_focus_relative"] = np.sort(
        set_relative(clean["question_focus"], clean["groups"])
    )

    known = settings["kind1"][0]

    def method_scores(data):
        scores = {
            method: sign * data[feature]
            for method, (feature, sign) in SINGLE_METHODS.items()
        }
        scores["question_focus_relative"] = set_relative(
            data["question_focus"], data["groups"]
        )

        for method, features in TRAINED_METHODS.items():
            scores[method] = trained_on_known_attack(
                np.stack([known[f] for f in features], axis=1),
                known["labels"], known["groups"],
                np.stack([data[f] for f in features], axis=1),
                data["groups"],
            )

        fractions = {
            m: np.searchsorted(reference[m], scores[m], side="right")
            / len(reference[m])
            for m in {p for parts in RANK_COMBINATIONS.values()
                      for p in parts}
        }

        for suffix, parts in RANK_COMBINATIONS.items():
            chosen = np.stack([fractions[m] for m in parts])
            scores[f"rank_mean{suffix}"] = chosen.mean(axis=0)
            scores[f"rank_max{suffix}"] = chosen.max(axis=0)

        return scores

    for draws in settings.values():
        for data in draws:
            data["scores"] = method_scores(data)

    methods = list(settings["kind1"][0]["scores"])

    def setting_aurocs(draws, samples=None):
        """Mean AUROC over the draws of a setting, for each method."""
        values = {m: [] for m in methods}

        for data in draws:
            index = (slice(None) if samples is None else np.concatenate(
                [data["rows_of"][q] for q in samples]
            ))

            for m in methods:
                values[m].append(
                    auroc(data["labels"][index], data["scores"][m][index])
                )

        return {m: float(np.mean(v)) for m, v in values.items()}

    results = {"settings": {}, "methods": {}}

    for name, draws in settings.items():
        aurocs = setting_aurocs(draws)
        results["settings"][name] = {
            m: {
                "auroc": aurocs[m],
                "tpr_at_5pct_fpr": float(np.mean([
                    tpr_at_fpr(d["labels"], d["scores"][m]) for d in draws
                ])),
            }
            for m in methods
        }

    for m in methods:
        values = {s: results["settings"][s][m]["auroc"] for s in settings}
        worst = min(values, key=values.get)
        results["methods"][m] = {
            "worst_auroc": values[worst],
            "worst_setting": worst,
            "mean_auroc": float(np.mean(list(values.values()))),
        }

    # Paired bootstrap over questions (the same questions in every
    # setting) for the worst-case AUROC.
    questions = sorted(set(known["groups"]))

    for draws in settings.values():
        for data in draws:
            data["rows_of"] = {
                q: np.flatnonzero(data["groups"] == q) for q in questions
            }

    boot_rng = np.random.default_rng(SEED)
    worst = {m: [] for m in methods}
    by_setting = {s: {m: [] for m in methods} for s in settings}

    for b in range(N_BOOTSTRAPS):
        samples = boot_rng.choice(questions, len(questions))
        per_setting = [setting_aurocs(d, samples) for d in settings.values()]

        for m in methods:
            worst[m].append(min(s[m] for s in per_setting))

            for name, values in zip(settings, per_setting):
                by_setting[name][m].append(values[m])

        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{N_BOOTSTRAPS}", flush=True)

    for m in methods:
        low, high = np.percentile(worst[m], [2.5, 97.5])
        results["methods"][m]["worst_auroc_ci_95"] = [float(low), float(high)]

    results["comparisons"] = {}

    for first, second in COMPARISONS:
        differences = np.array(worst[first]) - np.array(worst[second])
        low, high = np.percentile(differences, [2.5, 97.5])
        p_value = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
        results["comparisons"][f"{first} - {second} (worst AUROC)"] = {
            "auroc_difference": (results["methods"][first]["worst_auroc"]
                                 - results["methods"][second]["worst_auroc"]),
            "ci_95": [float(low), float(high)],
            "p_value": float(min(1.0, p_value)),
        }

    results["setting_comparisons"] = {}

    for name in settings:
        results["setting_comparisons"][name] = {}

        for first, second in SETTING_COMPARISONS:
            differences = (np.array(by_setting[name][first])
                           - np.array(by_setting[name][second]))
            low, high = np.percentile(differences, [2.5, 97.5])
            p_value = 2 * min((differences <= 0).mean(),
                              (differences >= 0).mean())
            results["setting_comparisons"][name][f"{first} - {second}"] = {
                "auroc_difference": (
                    results["settings"][name][first]["auroc"]
                    - results["settings"][name][second]["auroc"]
                ),
                "ci_95": [float(low), float(high)],
                "p_value": float(min(1.0, p_value)),
            }

    results["setup"] = {
        "settings": list(settings),
        "one_poisoned_draws": N_DRAWS,
        "trained_on": "kind1",
        "rank_parts": {f"rank{k}": list(v)
                       for k, v in RANK_COMBINATIONS.items()},
        "bootstraps": N_BOOTSTRAPS,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    short = {"kind1": "K1", "kind2": "K2", "kind1_one_poisoned": "K1 n=1",
             "kind2_one_poisoned": "K2 n=1", "reworded": "reword",
             "diverse": "diverse"}
    print("\nROBUSTNESS: AUROC IN EVERY SETTING (all benign negatives)")
    print("=" * 108)
    print(f"{'method':<26}" + "".join(f"{short[s]:>9}" for s in settings)
          + f"{'worst':>9}{'95% CI':>17}{'mean':>8}")

    for m in sorted(methods,
                    key=lambda m: -results["methods"][m]["worst_auroc"]):
        r = results["methods"][m]
        print(f"{m:<26}" + "".join(
            f"{results['settings'][s][m]['auroc']:>9.3f}" for s in settings
        ) + f"{r['worst_auroc']:>9.3f}"
            f"   [{r['worst_auroc_ci_95'][0]:.3f}, "
            f"{r['worst_auroc_ci_95'][1]:.3f}]{r['mean_auroc']:>8.3f}")

    print("\nWorst-case AUROC differences (paired bootstrap over questions)")

    for name, c in results["comparisons"].items():
        print(f"{name:<56}{c['auroc_difference']:>+8.3f}"
              f"   [{c['ci_95'][0]:+.3f}, {c['ci_95'][1]:+.3f}]"
              f"   p={c['p_value']:.3f}")

    print("\nInside each setting")

    for name, tests in results["setting_comparisons"].items():
        for test, c in tests.items():
            print(f"{short[name]:<9}{test:<47}{c['auroc_difference']:>+8.3f}"
                  f"   [{c['ci_95'][0]:+.3f}, {c['ci_95'][1]:+.3f}]"
                  f"   p={c['p_value']:.3f}")

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
