# Passage Trust Scoring against PoisonedRAG

This repository builds on [PoisonedRAG](https://arxiv.org/abs/2402.07867) (Zou et al., USENIX Security 2025), the knowledge-corruption attack on retrieval-augmented generation (RAG). The attack code and the original instructions come from the [PoisonedRAG authors](#acknowledgement). This project adds two extensions on the Natural Questions (NQ) setting:

- **[Passage Trust Scoring](#passage-trust-scoring)**: a defence that scores each retrieved passage for how likely it is to be poisoned, so that suspicious passages can be removed before the reader sees them.
- **[Paraphrase Robustness Experiment](#paraphrase-robustness-experiment)**: a study of whether rewording a question changes what the retriever finds.

The project is complete. All results are in `data_processed/` and all figures in `figures/`, and every step can be rerun with the scripts listed below. The original PoisonedRAG usage is kept in [Original PoisonedRAG usage](#original-poisonedrag-usage).

## Passage Trust Scoring

### Overview

This extension is a defence, not a new attack. It scores every retrieved passage for how likely it is to be poisoned, so that suspicious passages can be removed before the reader sees them. It uses 100 NQ target questions, a candidate pool of 9,872 passages and PoisonedRAG's black-box poisoned passages, in two forms: with the question prefix and without it. The signals tested include relevance consistency under question paraphrase, reverse questions (doc2query), the retrieval score, set density, shortness, GPT-2 perplexity, reader support, and combinations of these. Results are reported as AUROC, AUPRC and TPR at 5% FPR, the worst case over six attack settings, the end-to-end attack success rate with an extractive reader and with Qwen2.5-1.5B-Instruct, and the cost per question.

### Setup

The scripts were tested with Python 3.14 in a local virtual environment named `.venv`:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements_trust.txt
```

Contriever always runs on the CPU, even when a GPU or Apple MPS device is present, because MPS produced wrong embeddings for some batches. The other models (doc2query, GPT-2, the readers, the NLI model) use MPS when it is available. All models are downloaded from Hugging Face on the first run.

The trained combinations use `GroupKFold`, whose folds changed in scikit-learn 1.6. To reproduce the saved combination results exactly, install `scikit-learn<1.6` (1.5.2 was checked); newer versions shift some values slightly.

### Running the pipeline

`run_trust_pipeline.sh` runs every step in order, from the files in `data_processed/`. It downloads the NQ qrels to `datasets/nq/qrels/test.tsv` if they are missing.

```bash
bash run_trust_pipeline.sh            # uses .venv/bin/python
bash run_trust_pipeline.sh python3    # or another interpreter
```

The steps, in the order the script runs them:

| Step | Scripts |
|---|---|
| 1. Evaluation set and features | `build_trust_eval_set.py`, `generate_trust_embeddings.py`, `generate_answer_support.py`, `generate_perplexity.py`, `compute_trust_scores.py` |
| 2. Detection results | `evaluate_detection.py`, `evaluate_poison_count.py`, `analyse_failure_cases.py`, `analyse_why_consistency_fails.py` |
| 3. Paraphrase quality | `check_paraphrases_nli.py`, `evaluate_paraphrase_quality.py`, `compare_paraphrase_checks.py` |
| 4. End-to-end effect, reworded attack, combinations | `evaluate_end_to_end.py`, `evaluate_end_to_end_llm.py`, `evaluate_adaptive_attack.py`, `evaluate_combinations.py` |
| 5. Reverse questions | `generate_reverse_questions.py`, `generate_pool_focus.py`, `evaluate_reverse_questions.py`, `analyse_reverse_questions.py`, `evaluate_reverse_end_to_end.py`, `evaluate_diverse_attack.py`, `evaluate_robustness.py` |
| 6. Statistics and design-choice checks | `apply_holm_correction.py`, `evaluate_sensitivity.py` |
| 7. Running time and figures | `measure_cost.py`, `make_figures.py` |

Results are written to `data_processed/trust/` and figures to `figures/`. `measure_cost.py` times the methods, so its numbers depend on the machine.

**Do not run `make_paraphrase_review_files.py`.** It rewrites the manual paraphrase review files and would replace the reviews that were filled in by hand. The pipeline does not run it.

### Report numbers, tables and figures

`make_report_numbers.py` writes every reported number as a LaTeX macro and builds the report tables, so no result is typed by hand. It runs after the pipeline:

```bash
.venv/bin/python make_report_numbers.py
```

`make_figures.py` writes each figure to `figures/<name>.pdf` (and a PNG); `report/figures/` holds copies of those PDFs. Unless noted, inputs are in `data_processed/trust/`.

The top-30 variant (`question_focus_relative_top30`) comes from `generate_pool_focus.py`, which writes `pool_reverse_questions.csv` and `pool_question_focus.csv`. `evaluate_robustness.py` and `evaluate_reverse_end_to_end.py` add its rows to `robustness_results.json` and `reverse_end_to_end_results.json`, so it reaches `report/numbers.tex` and `report/tables/` through those files.

| Report file | Made by | Input |
|---|---|---|
| `report/numbers.tex` | `make_report_numbers.py` | `robustness_results.json`, `holm_corrected_pvalues.json`, `detection_results.json`, `reverse_end_to_end_results.json`, `cost_results.json`, and `data_processed/experiment_retrieval_metrics.json`, `data_processed/baseline_retrieval_metrics.json`, `data_processed/paraphrase_significance_results.json` |
| `report/tables/robustness.tex` | `make_report_numbers.py` | `robustness_results.json`, `holm_corrected_pvalues.json` |
| `report/tables/detection.tex` | `make_report_numbers.py` | `detection_results.json` |
| `report/tables/end_to_end.tex` | `make_report_numbers.py` | `reverse_end_to_end_results.json` |
| `report/tables/cost.tex` | `make_report_numbers.py` | `cost_results.json` |
| `report/tables/paraphrase.tex` | `make_report_numbers.py` | `data_processed/experiment_retrieval_metrics.json`, `data_processed/baseline_retrieval_metrics.json`, `data_processed/paraphrase_significance_results.json` |
| `report/figures/detection_auroc.pdf` | `make_figures.py` | `detection_results.json` |
| `report/figures/poison_count.pdf` | `make_figures.py` | `poison_count_results.json` |
| `report/figures/cost_vs_detection.pdf` | `make_figures.py` | `detection_results.json`, `reverse_question_results.json`, `cost_results.json` |
| `report/figures/z_drop_distribution.pdf` | `make_figures.py` | `trust_scores.csv`, `failure_analysis.json` |
| `report/figures/roc_curves.pdf` | `make_figures.py` | `trust_scores.csv`, `detection_results.json` |
| `report/figures/end_to_end.pdf` | `make_figures.py` | `reverse_end_to_end_results.json` |
| `report/figures/adaptive_attack.pdf` | `make_figures.py` | `adaptive_attack_results.json` |
| `report/figures/combinations_poison_count.pdf` | `make_figures.py` | `combination_results.json` |
| `report/figures/z_drop_by_level.pdf` | `make_figures.py` | `why_consistency_fails.json` |
| `report/figures/reverse_questions.pdf` | `make_figures.py` | `reverse_question_results.json`, `reverse_question_analysis.json` |
| `report/figures/robustness.pdf` | `make_figures.py` | `robustness_results.json` |

## Paraphrase Robustness Experiment

### Overview

This experiment tests whether rewording a question changes what the retriever finds. Each of the 100 NQ target questions has three paraphrases, and Contriever retrieval for the original questions is compared with retrieval for their paraphrases over the same 9,872-passage candidate pool. It asks three questions:

1. Do paraphrased queries find relevant passages more or less often than the original questions?
2. How consistent is retrieval across different wordings of the same question?
3. Are the differences statistically significant?

### Queries and paraphrase validation

| Component | Count |
|---|---:|
| Original questions | 100 |
| Paraphrases per question | 3 |
| Total paraphrases | 300 |
| Total experiment queries | 400 |

Each paraphrase was compared with its original question using cosine similarity from `sentence-transformers/all-MiniLM-L6-v2`.

| Check | Threshold | Passed | Flagged for review |
|---|---:|---:|---:|
| First pass (generated paraphrases) | 0.80 | 212 | 88 |
| Corrected paraphrases (used in the experiment) | 0.75 | 280 | 20 |

The flagged paraphrases from the first pass were corrected. In the corrected set, the 20 flagged paraphrases come from 14 questions. A flag means the paraphrase needs a closer reading, not that it is wrong. Not all 300 paraphrases were reviewed by hand.

- `data_processed/paraphrase_validation.csv`: first pass.
- `data_processed/corrected_paraphrase_validation.csv`: corrected set.
- `data_processed/final_paraphrase_summary.json`: counts for the corrected set.

`validate_paraphrases.py` uses the 0.75 threshold but reads the generated paraphrases, not the corrected set.

### Embeddings

| Parameter | Value |
|---|---|
| Retrieval model | `facebook/contriever` |
| Maximum sequence length | 512 tokens |
| Batch size | 8 |
| Embedding dimension | 768 |
| Candidate passages | 9,872 |
| Queries | 400 (100 original, 300 paraphrased) |
| Retrieval depth | Top 100 |
| Similarity scoring | Dot product |

An earlier run with a 128-token limit truncated passages and gave an original-query Hit@10 of 25%. With 512 tokens, the original-query results match the baseline in `data_processed/baseline_retrieval_metrics.json` on all seven metrics. The embedding files were checked for shape (400 × 768 queries, 9,872 × 768 passages), aligned and unique query IDs, and NaN or infinite values; all checks passed.

### Metrics

Relevance comes from the NQ qrels (`datasets/nq/qrels/test.tsv`).

- **Hit@k** (k = 1, 5, 10, 100): share of queries with at least one relevant passage in the top k.
- **Recall@k** (k = 10, 100): mean share of a query's relevant passages found in the top k.
- **MRR@10**: mean reciprocal rank of the first relevant passage in the top 10.

### Results

| Metric | Original queries (100) | Paraphrased queries (300) |
|---|---:|---:|
| Hit@1 | 0.0600 | 0.0567 |
| Hit@5 | 0.1800 | 0.1900 |
| Hit@10 | 0.3100 | 0.3067 |
| Hit@100 | 0.7900 | 0.8900 |
| Recall@10 | 0.2783 | 0.2833 |
| Recall@100 | 0.7667 | 0.8700 |
| MRR@10 | 0.1142 | 0.1188 |

For the paired comparison, each original question is matched with the mean score of its three paraphrases (difference = paraphrase mean − original). Significance uses a paired permutation test and a 95% bootstrap confidence interval.

| Metric | Improved | Worsened | Unchanged | Mean difference | p-value | 95% CI | Significant (p < 0.05) |
|---|---:|---:|---:|---:|---:|---|---|
| Hit@10 | 17 | 16 | 67 | −0.0033 | 1.0000 | [−0.0900, 0.0833] | No |
| Hit@100 | 15 | 2 | 83 | +0.1000 | 0.0016 | [0.0400, 0.1633] | Yes |
| MRR@10 | 24 | 22 | 54 | +0.0047 | 0.8110 | [−0.0311, 0.0425] | No |

Rewording does not change top-10 retrieval in any significant way, and most questions keep the same outcome. Paraphrases do find a relevant passage within the top 100 more often (+10 percentage points, p = 0.0016). So paraphrasing mainly moves where relevant passages rank, rather than making retrieval better or worse overall. Some paraphrases may not keep the exact meaning of the question, so this effect should not be read as a general benefit of paraphrasing.

### Limitations

1. **Sample size:** 100 questions and 300 paraphrases limit statistical power.
2. **Meaning preservation:** embedding similarity does not guarantee that a paraphrase means the same as the question, and not every paraphrase was checked by hand.
3. **Retrieval only:** the experiment measures retrieval, not the quality of the final answer.
4. **Restricted pool:** retrieval runs over 9,872 candidate passages, not the full NQ corpus.
5. **Baseline provenance:** the 512-token results reproduce the baseline metrics, but the configuration that produced that baseline was not independently confirmed.
6. **One setting:** results may differ for other retrievers, datasets and embedding settings.

### Scripts

| Script | Purpose |
|---|---|
| `generate_reworded_questions.py` | Generate the paraphrases |
| `validate_paraphrases.py` | Score paraphrase similarity |
| `reconstruct_candidate_passages.py` | Rebuild the 9,872 candidate passages from their IDs and the NQ corpus |
| `generate_passage_embeddings.py` | Encode passages with Contriever |
| `generate_query_embeddings.py` | Encode queries with Contriever |
| `run_paraphrase_retrieval.py` | Run retrieval for all 400 queries |
| `evaluate_experiment_retrieval.py` | Compute Hit@k, Recall@k and MRR@10 |
| `evaluate_paraphrase_robustness.py` | Compute the paired differences |
| `evaluate_paraphrase_significance.py` | Run the significance tests |

Inputs, embedding metadata and results are in `data_processed/`.

## Original PoisonedRAG usage

![Illustration of PoisonedRAG](PoisonedRAG.png "Illustration of PoisonedRAG")

### Setup environment

```bash
conda create -n PoisonedRAG python=3.10
```
```bash
conda activate PoisonedRAG
```
```bash
pip install beir openai google-generativeai
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade charset-normalizer
pip3 install "fschat[model_worker,webui]"
```

### Dataset (optional)

The datasets are downloaded to `datasets/` automatically on first use. To download them in advance, run:

```bash
python prepare_dataset.py
```

### Set API key

To use PaLM 2, GPT-3.5, GPT-4 or LLaMA-2, add the API key to the matching file in `model_configs/`.

For LLaMA-2, the key is a Hugging Face access token; request access on [LLaMA-2's HuggingFace Page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) if you do not have one.

For example:

```json
"api_key_info":{
    "api_keys":[
        "Your api key here"
    ],
    "api_key_use": 0
},
```

### Reproduce the paper's results

`run.py` sets the LLM, dataset and attack settings. The defaults reproduce the main results of the paper.

```python
test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",            # nq, hotpotqa, msmarco
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'palm2',           # palm2, gpt3.5, gpt4, llama(7b|13b), vicuna(7b|13b|33b)
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',  # LM_targeted (black-box), hotflip (white-box)
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,

    'note': None
}
```

Then run:

```bash
python run.py
```

### Your own dataset
To use another dataset, convert it to the BEIR format and compute retrieval scores with `evaluate_beir.py`.

## Acknowledgement

* The attack code and original instructions are from [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) by Zou, Geng, Wang and Jia.
* PoisonedRAG builds on [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning), takes its model wrappers from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection), and uses the [BEIR](https://github.com/beir-cellar/beir) benchmark and [Contriever](https://github.com/facebookresearch/contriever) for retrieval.

## Citation

If you use this code, please cite the PoisonedRAG [paper](https://arxiv.org/abs/2402.07867):

```bib
@inproceedings{zou2025poisonedrag,
  title={$\{$PoisonedRAG$\}$: Knowledge corruption attacks to $\{$Retrieval-Augmented$\}$ generation of large language models},
  author={Zou, Wei and Geng, Runpeng and Wang, Binghui and Jia, Jinyuan},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={3827--3844},
  year={2025}
}
```

