
import json
from pathlib import Path

from evaluate_detection import N_BOOTSTRAPS

# Holm correction for the paired bootstrap tests. Each experiment is one
# family of tests. A bootstrap p-value of 0 only means that no resample
# crossed zero, so it is replaced by the smallest value the test can give,
# 2 / (N_BOOTSTRAPS + 1) (two-sided), before correcting.
TRUST_DIR = Path("data_processed/trust")
OUTPUT_PATH = TRUST_DIR / "holm_corrected_pvalues.json"

FAMILIES = {
    "detection": "detection_results.json",
    "combinations": "combination_results.json",
    "paraphrase_quality": "paraphrase_quality_results.json",
}
ALPHA = 0.05
SMALLEST_P = 2 / (N_BOOTSTRAPS + 1)


def find_tests(node, path=""):
    """All dictionaries with a p_value inside a results file."""
    if not isinstance(node, dict):
        return []

    if "p_value" in node:
        return [(path, node)]

    return [
        test for key, value in node.items()
        for test in find_tests(value, f"{path}/{key}" if path else key)
    ]


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0

    for step, i in enumerate(order):
        running = max(running, (len(p_values) - step) * p_values[i])
        adjusted[i] = min(1.0, running)

    return adjusted


def main():
    results = {"alpha": ALPHA, "smallest_p": SMALLEST_P, "families": {}}

    for family, name in FAMILIES.items():
        with (TRUST_DIR / name).open(encoding="utf-8") as f:
            tests = find_tests(json.load(f))

        raw = [max(t["p_value"], SMALLEST_P) for _, t in tests]
        adjusted = holm(raw)

        results["families"][family] = [
            {
                "test": path,
                "difference": t.get("auroc_difference", t.get("difference")),
                "p_value": p,
                "p_holm": q,
                "significant": q < ALPHA,
            }
            for (path, t), p, q in zip(tests, raw, adjusted)
        ]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nHOLM-CORRECTED P-VALUES (one family per experiment)")
    print("=" * 96)

    for family, tests in results["families"].items():
        print(f"\n{family} ({len(tests)} tests)")

        for t in tests:
            difference = (f"{t['difference']:+.3f}"
                          if t["difference"] is not None else "")
            print(f"  {t['test']:<70}{difference:>8}"
                  f"  p={t['p_value']:.3f}  holm={t['p_holm']:.3f}"
                  f"{'  *' if t['significant'] else ''}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
