
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Draws the report figures from the saved result files. No numbers are
# computed here; every value comes from data_processed/trust/.
TRUST_DIR = Path("data_processed/trust")
FIGURE_DIR = Path("figures")

# IEEE two-column paper: one column is about 3.5 inches wide.
COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.16

LABELS = {
    "retrieval_z": "Retrieval score (z)",
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
    "retrieval_z", "set_density", "shortness", "perplexity", "reader_support",
)
ATTACK_LABELS = {
    "prefix": "With question prefix",
    "noprefix": "Without question prefix",
}
ATTACK_COLOURS = {"prefix": "#0072B2", "noprefix": "#E69F00"}
METHOD_STYLES = {
    "retrieval_z": ("#000000", "s", "-"),
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
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
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
        "retrieval_z", "set_density", "z_drop_k3", "combined_consistency",
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


def main():
    detection_figure()
    poison_count_figure()
    cost_figure()
    distribution_figure()


if __name__ == "__main__":
    main()
