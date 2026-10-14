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

# 3. Paraphrase quality.
run check_paraphrases_nli.py
run evaluate_paraphrase_quality.py
run compare_paraphrase_checks.py

# 4. End-to-end effect, reworded attack, combinations.
run evaluate_end_to_end.py
run evaluate_adaptive_attack.py
run evaluate_combinations.py

# 5. Running time (depends on the machine) and figures.
run measure_cost.py
run make_figures.py
