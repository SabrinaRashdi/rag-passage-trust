#!/usr/bin/env bash
# Runs all trust-scoring experiments in order, from the files in
# data_processed/ (dataset, candidate pool, paraphrases, manual reviews).
# Usage: bash run_trust_pipeline.sh [python]   (default: .venv/bin/python)
# Models are downloaded from Hugging Face on the first run.
# make_paraphrase_review_files.py is not run: it would replace the
# filled manual review files.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${1:-.venv/bin/python}"
QRELS=datasets/nq/qrels/test.tsv

if [ ! -f "$QRELS" ]; then
    mkdir -p "$(dirname "$QRELS")"
    curl -sL --max-time 300 -o "$QRELS" \
        https://huggingface.co/datasets/BeIR/nq-qrels/resolve/main/test.tsv
fi

run() {
    echo
    echo "################ $1"
    "$PYTHON" "$1"
}

# 1. Evaluation set and features.
run build_trust_eval_set.py
run generate_trust_embeddings.py
run generate_answer_support.py
run generate_perplexity.py
run compute_trust_scores.py

# 2. Detection results.
run evaluate_detection.py
run evaluate_poison_count.py
run analyse_failure_cases.py
run analyse_why_consistency_fails.py

# 3. Paraphrase quality.
run check_paraphrases_nli.py
run evaluate_paraphrase_quality.py
run compare_paraphrase_checks.py

# 4. End-to-end effect, reworded attack, combinations.
run evaluate_end_to_end.py
run evaluate_end_to_end_llm.py
run evaluate_adaptive_attack.py
run evaluate_combinations.py

# 5. Reverse questions (added after the main results).
run generate_reverse_questions.py
run evaluate_reverse_questions.py
run analyse_reverse_questions.py
run evaluate_reverse_end_to_end.py
run evaluate_diverse_attack.py
run evaluate_robustness.py

# 6. Statistics and design-choice checks.
run apply_holm_correction.py
run evaluate_sensitivity.py

# 7. Running time (depends on the machine) and figures.
run measure_cost.py
run make_figures.py
