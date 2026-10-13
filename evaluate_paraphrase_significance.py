
import json
from pathlib import Path

import numpy as np

INPUT_PATH = Path("data_processed/paraphrase_robustness_analysis.json")
OUTPUT_PATH = Path("data_processed/paraphrase_significance_results.json")

METRICS = ("hit10", "hit100", "mrr10")
N_PERMUTATIONS = 10000
N_BOOTSTRAPS = 10000
SEED = 42


def main():
    with INPUT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = data["per_question"]
    rng = np.random.default_rng(SEED)

    print("\nPAIRED STATISTICAL SIGNIFICANCE TEST")
    print("=" * 45)

    results = {}

    for metric in METRICS:
        differences = np.array([
            row[metric]["paraphrase_mean"]
            - row[metric]["original"]
            for row in rows
        ], dtype=float)

        observed = float(differences.mean())

        # Paired sign-flip permutation test
        extreme = 0

        for _ in range(N_PERMUTATIONS):
            signs = rng.choice([-1, 1], size=len(differences))
            permuted = np.mean(differences * signs)

            if abs(permuted) >= abs(observed) - 1e-12:
                extreme += 1

        p_value = (extreme + 1) / (N_PERMUTATIONS + 1)

        # Paired bootstrap confidence interval
        bootstrap = np.empty(N_BOOTSTRAPS)

        for i in range(N_BOOTSTRAPS):
            sample = rng.choice(
                differences,
                size=len(differences),
                replace=True
            )
            bootstrap[i] = sample.mean()

        ci_low, ci_high = np.percentile(
            bootstrap, [2.5, 97.5]
        )

        results[metric] = {
            "mean_difference": observed,
            "p_value": float(p_value),
            "ci_95": [float(ci_low), float(ci_high)],
            "significant_at_0_05": bool(p_value < 0.05)
        }

        print(f"\nMetric: {metric}")
        print(f"Questions: {len(differences)}")
        print(f"Mean difference: {observed:.4f}")
        print(f"Permutation p-value: {p_value:.5f}")
        print(
            f"95% bootstrap CI: "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )
        print(f"Significant at 0.05: {p_value < 0.05}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", OUTPUT_PATH)
    print("Statistical significance testing complete.")


if __name__ == "__main__":
    main()
