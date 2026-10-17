# PoisonedRAG

Official repo of [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25) paper: [PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2402.07867).

The first knowledge database corruption attack against Retrievals-Augmented Generation (RAG) system.

## 🚀 News
🎉 Jun 20, 2024: **PoisonedRAG gets accepted to [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25)!**

🔥 Apr 20, 2024: **If you have any question or need other code or data, feel free to open an issue or email us!**

![Illustration of PoisonedRAG](PoisonedRAG.png "Illustration of PoisonedRAG")

## 🔍 Quick Usage

### 📃 Setup environment

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

### 💽 Dataset (optional, suggested)

When running our code, the datasets will be automatically downloaded and saved in `datasets`. You could also run this line to manually download datasets.

```bash
python prepare_dataset.py
```

### 🔑 Set API key

If you want to use PaLM 2, GPT-3.5, GPT-4 or LLaMA-2, please enter your api key in **model_configs** folder.

For LLaMA-2, the api key is your **HuggingFace Access Tokens**. You could visit [LLaMA-2's HuggingFace Page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) first if you don't have the access token.

Here is an example:

```json
"api_key_info":{
    "api_keys":[
        "Your api key here"
    ],
    "api_key_use": 0
},
```

### 📝 Reproduce our results
 
There are some hyperparameters in **run.py** such as LLMs and datasets:

**Note:** Currently we provide default setting for main results in our [paper](https://arxiv.org/abs/2402.07867). We will update and complete other settings later.

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

Execute **run.py** to reproduce experiments.

```bash
python run.py
```

### 🐱 Your own dataset
If you want to perform experiments on your own dataset, you could refer to **evaluate_beir.py**, prepare your dataset as the beir format and use this file to compute the retrieval scores.


## 🛡️ Passage Trust Scoring (Project Extension)

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

## 🔬 Paraphrase Robustness Experiment (Project Extension)

### Overview

This project extends the original PoisonedRAG repository by investigating the robustness of retrieval performance under query paraphrasing.

The experiment examines whether semantically equivalent reformulations of a question produce consistent retrieval results. We compare retrieval performance between original questions and their paraphrased versions.

### Research Objectives

The experiment aims to investigate:

1. Whether paraphrased queries improve or reduce retrieval performance compared with original questions.
2. How consistently a retrieval system performs across different formulations of the same question.
3. Whether observed retrieval performance differences are statistically significant.

### Dataset and Query Preparation

The experiment uses 100 original questions from the Natural Questions (NQ) dataset.

Three paraphrased versions were prepared for each original question.

| Component | Count |
|---|---:|
| Original questions | 100 |
| Paraphrases per question | 3 |
| Total paraphrases | 300 |
| Total experiment queries | 400 |

Paraphrase similarity was evaluated using the `sentence-transformers/all-MiniLM-L6-v2` model.

Cosine similarity was calculated between each original question and its paraphrased versions to identify potentially inconsistent reformulations.

Paraphrases with low similarity scores were flagged for review. Corrections were subsequently made to address identified semantic inconsistencies.

The corrected paraphrase dataset was used in the final experiment.


### Paraphrase Validation

Semantic similarity was assessed using `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity was calculated between each original question and its paraphrased versions to identify potentially inconsistent reformulations.

#### Initial Validation

The initial validation used a similarity threshold of **0.80**.

| Validation Result | Count |
|---|---:|
| Total paraphrases | 300 |
| Automatic passes (≥ 0.80) | 212 |
| Flagged for review (< 0.80) | 88 |

#### Corrected Paraphrase Validation

Following corrections to the paraphrases, the corrected validation dataset was assessed using a similarity threshold of **0.75**.

| Validation Result | Count |
|---|---:|
| Total paraphrases | 300 |
| Automatic passes (≥ 0.75) | 280 |
| Flagged for review (< 0.75) | 20 |

The 20 flagged paraphrases correspond to 14 original questions.

Of these:

- 11 flagged paraphrases belong to the eight question IDs listed in the final review summary.
- 9 flagged paraphrases belong to six additional question IDs not included in the documented review list.

#### Final Validation Summary

The final operational similarity threshold was set to **0.75**.

An earlier summary reported 289 automatic passes and 11 manually accepted paraphrases. However, these figures are inconsistent with the corrected validation CSV and have been superseded by the verified counts of 280 automatic passes and 20 flagged paraphrases.

A `REVIEW` status indicates that the paraphrase requires further semantic inspection rather than automatically indicating an incorrect paraphrase.

Although the existing summary identifies eight question IDs associated with manual review, independent evidence confirming acceptance of all flagged paraphrases is unavailable. Therefore, the remaining flagged cases are not represented as automatically or manually validated.

A complete manual semantic audit of all 300 paraphrases was not performed.

#### Reproducibility Notes

- The initial validation results are stored in `data_processed/paraphrase_validation.csv`.
- The corrected validation results are stored in `data_processed/corrected_paraphrase_validation.csv`.
- The updated summary is stored in `data_processed/final_paraphrase_summary.json`.
- The original summary is preserved in `data_processed/final_paraphrase_summary_original.json`.
- The validation script, `validate_paraphrases.py`, uses a configurable similarity threshold of 0.75, but reads the original generated paraphrases rather than the corrected dataset.
- The corrected paraphrases were used for the retrieval robustness experiment.

These distinctions ensure that the original validation, corrected validation, and documented review stages are not incorrectly treated as identical.



### Embedding Generation and Validation

Query and passage embeddings were generated using the `facebook/contriever` model.

The final embedding configuration was:

| Parameter | Value |
|---|---|
| Retrieval model | `facebook/contriever` |
| Maximum sequence length | 512 tokens |
| Batch size | 8 |
| Embedding dimension | 768 |
| Candidate passages | 9,872 |
| Original questions | 100 |
| Paraphrased questions | 300 |
| Total experiment queries | 400 |
| Retrieval depth | Top 100 |
| Similarity scoring | Dot product |

Both passage and query embeddings were regenerated using `MAX_LENGTH = 512`.

The previous experiment used a maximum sequence length of 128 tokens. Increasing this limit to 512 improved original-query Hit@10 from 25% to 31%, matching the historical baseline result.

The original-query retrieval metrics also matched the historical baseline across all seven reported metrics, within rounding. This supports the hypothesis that passage truncation contributed to the earlier discrepancy, although the historical baseline configuration has not been independently verified.

The final embedding validation confirmed:

| Validation Check | Result |
|---|---|
| Total experiment queries | 400 |
| Query embedding shape | (400, 768) |
| Passage embedding shape | (9872, 768) |
| Query IDs match embedding IDs | True |
| Query ID ordering matches | True |
| Duplicate query IDs | 0 |
| NaN values | None |
| Infinite values | None |

All 300 paraphrases in the experiment query file matched the corrected paraphrase dataset.

These checks establish structural consistency between the query dataset, candidate passages, and embedding metadata.

### Retrieval Evaluation

Retrieval performance was evaluated using the Natural Questions relevance judgments (`datasets/nq/qrels/test.tsv`).

The evaluation used the same restricted candidate pool of 9,872 passages for both original and paraphrased queries.

The following metrics were calculated:

- **Hit@1:** Proportion of queries retrieving at least one relevant passage in the top 1 result.
- **Hit@5:** Proportion of queries retrieving at least one relevant passage in the top 5 results.
- **Hit@10:** Proportion of queries retrieving at least one relevant passage in the top 10 results.
- **Hit@100:** Proportion of queries retrieving at least one relevant passage in the top 100 results.
- **Recall@10:** Mean proportion of relevant passages retrieved within the top 10 results.
- **Recall@100:** Mean proportion of relevant passages retrieved within the top 100 results.
- **MRR@10:** Mean Reciprocal Rank of the first relevant passage within the top 10 results.

#### Retrieval Performance

| Metric | Original Queries (100) | Paraphrased Queries (300) |
|---|---:|---:|
| Hit@1 | 0.0600 | 0.0567 |
| Hit@5 | 0.1800 | 0.1900 |
| Hit@10 | 0.3100 | 0.3067 |
| Hit@100 | 0.7900 | 0.8900 |
| Recall@10 | 0.2783 | 0.2833 |
| Recall@100 | 0.7667 | 0.8700 |
| MRR@10 | 0.1142 | 0.1188 |

The original questions achieved a Hit@10 of 31.00%, compared with 30.67% for paraphrased questions.

Paraphrased questions achieved higher Hit@100 and Recall@100, although improvements were not consistent across all retrieval metrics.

### Experimental Results

The experiment compared retrieval performance for 100 original questions against the mean performance of their three paraphrases.

For each original question, the paired difference was calculated as:

`Mean Difference = Mean Paraphrase Score - Original Question Score`

| Metric | Improved | Worsened | Unchanged | Mean Difference |
|---|---:|---:|---:|---:|
| Hit@10 | 17 | 16 | 67 | -0.0033 |
| Hit@100 | 15 | 2 | 83 | +0.1000 |
| MRR@10 | 24 | 22 | 54 | +0.0047 |

Hit@10 showed a small negative average difference, indicating that paraphrasing did not improve top-10 retrieval performance overall.

Hit@100 showed a positive average difference of 0.1000, corresponding to an improvement of 10 percentage points.

Most questions showed unchanged retrieval performance, particularly for Hit@100.

### Statistical Significance Testing

Paired statistical significance testing was conducted across the 100 original questions.

Each original question was paired with the mean retrieval performance of its three paraphrases.

The analysis used:

- Paired permutation testing.
- Bootstrap confidence intervals.
- A significance threshold of 0.05.

| Metric | Mean Difference | Permutation p-value | 95% Bootstrap CI | Significant (p < 0.05) |
|---|---:|---:|---|---|
| Hit@10 | -0.0033 | 1.0000 | [-0.0900, 0.0833] | No |
| Hit@100 | +0.1000 | 0.0016 | [0.0400, 0.1633] | Yes |
| MRR@10 | +0.0047 | 0.8110 | [-0.0311, 0.0425] | No |

The Hit@10 and MRR@10 differences were not statistically significant at the 0.05 level.

However, the Hit@100 improvement was statistically significant (p = 0.0016), suggesting that paraphrased queries retrieved relevant passages within the top 100 more frequently in this experiment.

This result should not be interpreted as evidence that paraphrasing consistently improves retrieval quality, particularly because some paraphrases may not fully preserve the meaning of the original questions.

### Limitations

The experiment has several limitations:

1. **Dataset size:** Only 100 original questions and 300 paraphrases were evaluated, limiting statistical power and generalisability.
2. **Semantic similarity:** Embedding-based similarity scores do not guarantee semantic equivalence between original questions and paraphrases.
3. **Manual validation:** A complete manual semantic audit of all 300 paraphrases was not performed as part of the original validation process. Subsequent manual review identified cases requiring further correction.
4. **Retrieval-only evaluation:** The experiment evaluates passage retrieval rather than final generated answer quality.
5. **Restricted candidate pool:** Retrieval was evaluated over 9,872 candidate passages rather than the complete Natural Questions corpus.
6. **Historical baseline provenance:** Although the 512-token results reproduce the historical baseline metrics, the original baseline retrieval configuration has not been independently verified.
7. **Model and dataset dependence:** Results may vary across retrieval models, datasets, and embedding configurations.

### Reproducibility

The following scripts support the paraphrase robustness experiment.

| Script | Description |
|---|---|
| `generate_reworded_questions.py` | Generate paraphrased questions |
| `validate_paraphrases.py` | Evaluate paraphrase semantic similarity |
| `reconstruct_candidate_passages.py` | Reconstruct the 9,872 candidate passages from their IDs and the NQ corpus |
| `generate_passage_embeddings.py` | Generate passage embeddings using Contriever |
| `generate_query_embeddings.py` | Generate query embeddings using Contriever |
| `run_paraphrase_retrieval.py` | Execute paraphrase retrieval experiments |
| `evaluate_experiment_retrieval.py` | Calculate Hit@K, Recall@K, and MRR@10 |
| `evaluate_paraphrase_robustness.py` | Calculate paired retrieval performance differences |
| `evaluate_paraphrase_significance.py` | Perform statistical significance testing |

Experimental datasets, embedding metadata, validation reports, and evaluation results are stored in `data_processed/`.

The final retrieval experiment uses a maximum sequence length of 512 tokens for both passage and query embeddings.

### Conclusion

This experiment demonstrates that retrieval performance can vary across different formulations of the same question.

Using 512-token Contriever embeddings, original questions achieved a Hit@10 of 31.00%, while paraphrased questions achieved 30.67%. The difference was not statistically significant.

Paraphrased questions achieved a higher Hit@100 of 89.00%, compared with 79.00% for original questions. This improvement was statistically significant (p = 0.0016).

MRR@10 differences were also not statistically significant.

These findings indicate that paraphrasing can affect retrieval rankings and the depth at which relevant passages are found. However, improvements are metric-dependent, and semantic inconsistencies in paraphrases remain a potential confounding factor.

The experiment highlights the importance of evaluating retrieval robustness across multiple query formulations rather than relying exclusively on original question wording.

This extension focuses on query paraphrase robustness and should be distinguished from the original PoisonedRAG knowledge corruption attack evaluation.

---



## Acknowledgement

* Our code used the implementation of [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning).
* The model part of our code is from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
* Our code used [beir](https://github.com/beir-cellar/beir) benchmark.
* Our code used [contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG).



## Citation

If you use this code, please cite the following [paper](https://arxiv.org/abs/2402.07867):

```bib
@inproceedings{zou2025poisonedrag,
  title={$\{$PoisonedRAG$\}$: Knowledge corruption attacks to $\{$Retrieval-Augmented$\}$ generation of large language models},
  author={Zou, Wei and Geng, Runpeng and Wang, Binghui and Jia, Jinyuan},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={3827--3844},
  year={2025}
}
```

