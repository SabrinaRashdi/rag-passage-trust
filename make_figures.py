
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from evaluate_detection import cross_validated_scores, suspicion_scores

# Draws the report figures from the saved result files. No new results are
# computed here; every value comes from data_processed/trust/ (the ROC
# curves are drawn from the saved scores in trust_scores.csv, and their
# AUROC is checked against detection_results.json).
TRUST_DIR = Path("data_processed/trust")
FIGURE_DIR = Path("figures")

# IEEE two-column paper: one column is about 3.5 inches wide.
COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.16

LABELS = {
    "retrieval_score": "Retrieval score",
    "set_density": "Set density",
    "shortness": "Short passage length",
    "perplexity": "Perplexity",
    "reader_support": "Reader support",
    "z_drop_k3": "Relevance consistency",
    "support_drop_k3": "Support consistency",
    "answer_disagree_k3": "Answer consistency",
    "combined_consistency": "Combined consistency",
}
BASELINES = (
    "retrieval_score", "set_density", "shortness", "perplexity",
    "reader_support",
)
ATTACK_LABELS = {
    "prefix": "With question prefix",
    "noprefix": "Without question prefix",
}
ATTACK_COLOURS = {"prefix": "#0072B2", "noprefix": "#E69F00"}
METHOD_STYLES = {
    "retrieval_score": ("#000000", "s", "-"),
    "set_density": ("#D55E00", "o", "-"),
    "z_drop_k3": ("#0072B2", "^", "--"),
    "combined_consistency": ("#009E73", "D", "--"),
    "perplexity": ("#CC79A7", "v", ":"),
    "shortness": ("#E69F00", "P", ":"),
    "reader_support": ("#56B4E9", "X", ":"),
}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(name):
    with (TRUST_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def save(figure, name):
    FIGURE_DIR.mkdir(exist_ok=True)
    # No creation date, so that the same figure gives the same file.
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight",
                   metadata={"CreationDate": None})
    figure.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(figure)
    print("Saved:", FIGURE_DIR / f"{name}.pdf")


def detection_figure():
    """AUROC of every method for both attack versions, with 95% CIs."""
    results = load("detection_results.json")
    methods = list(LABELS)
    positions = np.arange(len(methods))[::-1]
    height = 0.38

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 3.2))

    for offset, attack in ((height / 2, "prefix"), (-height / 2, "noprefix")):
        scores = results[f"{attack}/all_benign"]["methods"]
        values = np.array([scores[m]["auroc"] for m in methods])
        low = np.array([scores[m]["auroc_ci_95"][0] for m in methods])
        high = np.array([scores[m]["auroc_ci_95"][1] for m in methods])

        axis.barh(
            positions + offset, values, height,
            xerr=[values - low, high - values],
            color=ATTACK_COLOURS[attack], label=ATTACK_LABELS[attack],
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
        )

    # 0.5 is what random guessing gives.
    axis.axvline(0.5, color="black", linewidth=0.7, linestyle="--")
    # Line between the baselines and the paraphrase-based methods.
    axis.axhline(
        positions[len(BASELINES) - 1] - 0.5, color="grey", linewidth=0.5
    )
    axis.set_yticks(positions)
    axis.set_yticklabels([LABELS[m] for m in methods])
    axis.set_xlim(0, 1)
    axis.set_xlabel("AUROC")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
    )
    save(figure, "detection_auroc")


def poison_count_figure():
    """AUROC against the number of poisoned passages per question."""
    results = load("poison_count_results.json")
    methods = (
        "retrieval_score", "set_density", "z_drop_k3", "combined_consistency",
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 1.9), sharey=True
    )

    for axis, (attack, by_count) in zip(axes, results.items()):
        counts = sorted(by_count, key=int)

        for method in methods:
            colour, marker, line = METHOD_STYLES[method]
            axis.plot(
                [int(c) for c in counts],
                [by_count[c][method]["auroc_mean"] for c in counts],
                color=colour, marker=marker, linestyle=line,
                markersize=3.5, linewidth=1, label=LABELS[method],
            )

        axis.axhline(0.5, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(ATTACK_LABELS[attack], fontsize=8)
        axis.set_xticks([int(c) for c in counts])
        axis.set_ylim(0.3, 1.02)

    axes[0].set_ylabel("AUROC")
    figure.supxlabel("Poisoned passages per question", fontsize=8, y=0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98),
        ncol=2, frameon=False,
    )
    figure.tight_layout()
    save(figure, "poison_count")


def cost_figure():
    """Detection quality against the extra time each method needs."""
    detection = load("detection_results.json")["prefix/all_benign"]["methods"]
    cost = load("cost_results.json")["extra_cost"]

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))

    for method, values in cost.items():
        colour, marker, _ = METHOD_STYLES[method]
        axis.scatter(
            values["median_ms"], detection[method]["auroc"],
            color=colour, marker=marker, s=28, label=LABELS[method],
            zorder=3,
        )

    # Linear below 1 ms so that methods with no extra cost can be shown.
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlim(-0.3, 20000)
    axis.set_xticks([0, 1, 10, 100, 1000, 10000])
    axis.set_xticklabels(["0", "1", "10", "100", "1,000", "10,000"])
    axis.set_ylim(0.5, 1.02)
    axis.set_xlabel("Extra time per question (ms)")
    axis.set_ylabel("AUROC (with question prefix)")
    axis.grid(linewidth=0.3, alpha=0.6)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
    )
    save(figure, "cost_vs_detection")


def distribution_figure():
    """How the relevance-consistency score is spread for each passage type."""
    with (TRUST_DIR / "trust_scores.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    threshold = load("failure_analysis.json")["threshold_z_drop"]
    attacks = np.array([r["attack"] for r in rows])
    z_drop = np.array([float(r["z_drop"]) for r in rows])
    bins = np.linspace(
        np.percentile(z_drop, 0.5), np.percentile(z_drop, 99.5), 32
    )

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))

    for attack, label, colour in (
        ("none", "Benign", "#999999"),
        ("prefix", "Poisoned, with prefix", ATTACK_COLOURS["prefix"]),
        ("noprefix", "Poisoned, without prefix", ATTACK_COLOURS["noprefix"]),
    ):
        axis.hist(
            z_drop[attacks == attack], bins=bins, density=True,
            histtype="step", linewidth=1.1, color=colour, label=label,
        )

    axis.axvline(threshold, color="black", linewidth=0.7, linestyle="--")
    axis.text(
        threshold, axis.get_ylim()[1], " 5% false alarms",
        fontsize=6.5, va="top", ha="left",
    )
    axis.set_xlabel("Relevance-consistency score (drop in z-score)")
    axis.set_ylabel("Density")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
    )
    save(figure, "z_drop_distribution")


def roc_figure():
    """ROC curves of the main methods for both attack versions."""
    with (TRUST_DIR / "trust_scores.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    text_columns = ("passage_id", "query_id", "attack")
    data = {
        key: np.array([
            r[key] if key in text_columns else float(r[key]) for r in rows
        ])
        for key in rows[0]
    }
    labels = data["label"].astype(int)
    signals = suspicion_scores(data, 3)
    detection = load("detection_results.json")
    methods = (
        "retrieval_score", "set_density", "reader_support", "z_drop_k3",
        "combined_consistency",
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 2.0), sharey=True
    )

    for axis, attack in zip(axes, ATTACK_LABELS):
        keep = (data["attack"] == "none") | (data["attack"] == attack)
        scores = {
            "retrieval_score": data["score_original"],
            "set_density": data[f"set_density_{attack}"],
            "reader_support": data["support_original"],
            "z_drop_k3": signals["z_drop_k3"],
        }
        combined = np.full(len(labels), np.nan)
        combined[keep] = cross_validated_scores(
            np.stack([signals[f][keep] for f in (
                "z_drop_k3", "support_drop_k3", "answer_disagree_k3",
            )], axis=1),
            labels[keep], data["query_id"][keep],
        )
        scores["combined_consistency"] = combined
        saved = detection[f"{attack}/all_benign"]["methods"]

        for method in methods:
            assert np.isclose(
                roc_auc_score(labels[keep], scores[method][keep]),
                saved[method]["auroc"],
            )
            fpr, tpr, _ = roc_curve(labels[keep], scores[method][keep])
            colour, _, line = METHOD_STYLES[method]
            axis.plot(fpr, tpr, color=colour, linestyle=line, linewidth=1,
                      label=LABELS[method])

        axis.plot([0, 1], [0, 1], color="black", linewidth=0.5,
                  linestyle=":")
        axis.set_title(ATTACK_LABELS[attack], fontsize=8)
        axis.set_xlabel("False-positive rate")
        axis.set_aspect("equal")

    axes[0].set_ylabel("True-positive rate")
    handles, labels_ = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels_, loc="lower center", bbox_to_anchor=(0.5, 0.98),
        ncol=2, frameon=False,
    )
    figure.tight_layout()
    save(figure, "roc_curves")


def end_to_end_figure():
    """Attack success with each defence, for both readers."""
    readers = (
        ("end_to_end_results.json", "Extractive reader"),
        ("end_to_end_llm_results.json", "Qwen2.5-1.5B-Instruct"),
    )
    methods = (
        "no_defence", "retrieval_score", "set_density", "shortness",
        "perplexity", "reader_support", "z_drop_k3", "combined_consistency",
        "remove_all_poison",
    )
    names = {**LABELS, "no_defence": "No defence",
             "remove_all_poison": "Remove all poison"}
    positions = np.arange(len(methods))[::-1]
    height = 0.38

    figure, axes = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 2.4), sharey=True
    )

    for axis, (name, title) in zip(axes, readers):
        results = load(name)

        for offset, attack in ((height / 2, "prefix"),
                               (-height / 2, "noprefix")):
            values = np.array([
                results[attack][m]["attacked"]["attack_success"]["rate"]
                for m in methods
            ])
            low = np.array([
                results[attack][m]["attacked"]["attack_success"]["ci_95"][0]
                for m in methods
            ])
            high = np.array([
                results[attack][m]["attacked"]["attack_success"]["ci_95"][1]
                for m in methods
            ])
            axis.barh(
                positions + offset, values, height,
                xerr=[values - low, high - values],
                color=ATTACK_COLOURS[attack], label=ATTACK_LABELS[attack],
                error_kw={"elinewidth": 0.7, "capsize": 1.5},
            )

        axis.set_yticks(positions)
        axis.set_yticklabels([names[m] for m in methods])
        axis.set_xlim(0, 1)
        axis.set_xlabel("Attack success rate")
        axis.set_title(title, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98),
        ncol=2, frameon=False,
    )
    figure.tight_layout()
    save(figure, "end_to_end")


def adaptive_figure():
    """AUROC on the known attack and on the reworded attack."""
    detection = load("adaptive_attack_results.json")["detection"]
    methods = (
        "retrieval_score", "set_density", "shortness", "perplexity",
        "reader_support", "z_drop_k3", "combined_consistency",
    )
    positions = np.arange(len(methods))[::-1]

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))

    for position, method in zip(positions, methods):
        before = detection[method]["auroc_known_attack"]
        after = detection[method]["auroc"]
        axis.annotate(
            "", xy=(after, position), xytext=(before, position),
            arrowprops={"arrowstyle": "->", "color": "grey",
                        "linewidth": 0.8},
        )

    axis.scatter(
        [detection[m]["auroc_known_attack"] for m in methods], positions,
        facecolors="none", edgecolors=ATTACK_COLOURS["prefix"], s=30,
        linewidths=1, zorder=4, label="Known attack (with prefix)",
    )
    axis.scatter(
        [detection[m]["auroc"] for m in methods], positions,
        color="#D55E00", marker="s", s=16, zorder=3,
        label="Reworded attack",
    )
    axis.axvline(0.5, color="black", linewidth=0.7, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels([LABELS[m] for m in methods])
    axis.set_xlim(0.3, 1.02)
    axis.set_xlabel("AUROC")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
    )
    save(figure, "adaptive_attack")


def combination_figure():
    """AUROC of the combined detectors against the number of poisoned
    passages per question."""
    results = load("combination_results.json")["poison_count"]
    styles = {
        "ours": ("Combined consistency (CC)", "#0072B2", "^", "--"),
        "set_density": ("Set density", "#D55E00", "o", "-"),
        "ours+set_density": ("CC + set density", "#CC79A7", "v", "--"),
        # Retrieval score, set density and reader support.
        "all_three_without_ours": ("Three baselines", "#000000", "s", "-"),
        "ours+all_three": ("CC + three baselines", "#009E73", "D", "--"),
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 1.9), sharey=True
    )

    for axis, attack in zip(axes, ATTACK_LABELS):
        by_count = results[attack]
        counts = sorted(by_count, key=int)

        for method, (label, colour, marker, line) in styles.items():
            axis.plot(
                [int(c) for c in counts],
                [by_count[c][method] for c in counts],
                color=colour, marker=marker, linestyle=line,
                markersize=3.5, linewidth=1, label=label,
            )

        axis.axhline(0.5, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(ATTACK_LABELS[attack], fontsize=8)
        axis.set_xticks([int(c) for c in counts])
        axis.set_ylim(0.3, 1.02)

    axes[0].set_ylabel("AUROC")
    figure.supxlabel("Poisoned passages per question", fontsize=8, y=0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98),
        ncol=2, frameon=False,
    )
    figure.tight_layout()
    save(figure, "combinations_poison_count")


def main():
    detection_figure()
    poison_count_figure()
    cost_figure()
    distribution_figure()
    roc_figure()
    end_to_end_figure()
    adaptive_figure()
    combination_figure()


if __name__ == "__main__":
    main()
