
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_detection import cross_validated_scores
from evaluate_end_to_end import (
    ANSWER_SUPPORT_PATH, defended_read_lists, print_results, score_answers,
)
from evaluate_end_to_end_llm import MODEL_NAME, llm_answers
from generate_pool_focus import load_pool_focus, reference_median
from generate_reverse_questions import SCORES_PATH

# End-to-end effect of the reverse-question defences, with the same
# retrieved lists, threshold rule and readers as evaluate_end_to_end.py and
# evaluate_end_to_end_llm.py. The other defences are run again so that all
# results in the output file come from the same run.
# Added later (post hoc, see evaluate_robustness.py): set-relative question
# focus, computed inside each retrieved list like set density, and
# rank_max (retrieval, set density, reader support, question focus), with
# each part compared with the passages of the clean retrieved lists.
# Then question_focus_relative_top30: the median is taken over the top 30
# retrieved passages instead of the read list (generate_pool_focus.py).
TRUST_DIR = Path("data_processed/trust")
OUTPUT_PATH = TRUST_DIR / "reverse_end_to_end_results.json"
TRUST_SCORES_PATH = TRUST_DIR / "trust_scores.csv"


def main():
    with SCORES_PATH.open(encoding="utf-8") as f:
        focus = {r["passage_id"]: float(r["question_focus"])
                 for r in csv.DictReader(f)}

    def extra_methods(attack, rows):
        attacks = np.array([r["attack"] for r in rows])
        keep = (attacks == "none") | (attacks == attack)
        labels = np.array([int(r["label"]) for r in rows])
        groups = np.array([r["query_id"] for r in rows])
        values = np.array([focus[r["passage_id"]] for r in rows])
        retrieval = np.array([float(r["score_original"]) for r in rows])

        # Trained on other questions only, as the combined defence.
        combined = np.full(len(rows), np.nan)
        combined[keep] = cross_validated_scores(
            np.stack([retrieval, values], axis=1)[keep],
            labels[keep], groups[keep],
        )
        return {"question_focus": values, "retrieval+focus": combined}

    def relative_focus(attack, condition, qid, pids):
        values = np.array([focus[pid] for pid in pids])
        return values - np.median(values)

    with TRUST_SCORES_PATH.open(encoding="utf-8") as f:
        poisoned = {}

        for r in csv.DictReader(f):
            if r["label"] == "1":
                poisoned.setdefault((r["attack"], r["query_id"]), []).append(
                    (float(r["score_original"]), focus[r["passage_id"]])
                )

    pool_focus = load_pool_focus()

    def relative_focus_top30(attack, condition, qid, pids):
        # The attacked corpus holds all 5 poisoned passages of the
        # question, whether or not they reach the read list.
        median = reference_median(
            pool_focus[qid],
            poisoned[attack, qid] if condition == "attacked" else [],
        )
        return np.array([focus[pid] for pid in pids]) - median

    dataset, passages, questions, runs, setup = defended_read_lists(
        extra_methods,
        list_methods={
            "question_focus_relative": relative_focus,
            "question_focus_relative_top30": relative_focus_top30,
        },
        rank_methods={"rank_max": (
            "retrieval_score", "set_density", "reader_support",
            "question_focus",
        )},
    )

    with ANSWER_SUPPORT_PATH.open(encoding="utf-8") as f:
        reader = {row["passage_id"]: row for row in csv.DictReader(f)}

    def extractive_answer(qid, read):
        if not read:
            return ""
        best = max(read, key=lambda pid: float(
            reader[pid]["support_original"]
        ))
        return reader[best]["answer_original"]

    results = {"setup": setup}

    for name, answer_of in (
        ("extractive_reader", extractive_answer),
        (MODEL_NAME, llm_answers(dataset, passages, runs)),
    ):
        results[name] = score_answers(
            dataset, passages, questions, runs, answer_of
        )
        print_results({**results[name], "setup": setup},
                      f"END-TO-END WITH REVERSE QUESTIONS ({name})")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
