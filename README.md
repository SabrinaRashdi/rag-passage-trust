# Passage Trust Scoring against PoisonedRAG

This repository extends [PoisonedRAG](https://arxiv.org/abs/2402.07867) (Zou et al., USENIX Security 2025), a knowledge-corruption attack on retrieval-augmented generation (RAG), with two studies on the Natural Questions (NQ) setting:

- **[Passage Trust Scoring](#passage-trust-scoring)**: a defence that scores each retrieved passage for how likely it is to be poisoned, so that suspicious passages can be removed before the reader sees them.
- **[Paraphrase Robustness](#paraphrase-robustness)**: a study of whether rewording a question changes what the retriever finds.

The attack code is from the [PoisonedRAG authors](#acknowledgement). Its original usage is described in [Original PoisonedRAG usage](#original-poisonedrag-usage).

## Contents

- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Passage Trust Scoring](#passage-trust-scoring)
- [Paraphrase Robustness](#paraphrase-robustness)
- [Original PoisonedRAG usage](#original-poisonedrag-usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Key results

**Passage Trust Scoring.** Detection is measured as the worst-case AUROC over six attack settings (see [Evaluation](#evaluation)).

| Method | Worst-case AUROC | Mean AUROC |
|---|---:|---:|
| Relevance consistency under paraphrase (`z_drop`) | 0.389 | 0.518 |
| Set density | 0.389 | 0.761 |
| GPT-2 perplexity | 0.680 | 0.718 |
| Reverse-question focus | 0.703 | 0.840 |
| Reverse-question focus, set-relative | 0.774 | 0.902 |
| Rank-max combination of baselines | 0.870 | 0.947 |
| Rank-max combination with focus | 0.871 | 0.952 |
| Rank-max combination with set-relative focus | 0.874 | 0.954 |

- Relevance consistency under paraphrase does not detect poisoned passages; in the worst case it ranks them below chance.
- The reverse-question signal is the strongest single signal in the worst case, but adds little on top of the rank-max combination of baselines.
- Signals with high average scores can fail under adaptive attacks: set density has a mean AUROC of 0.761 but falls to 0.389 in the worst case.
- With Qwen2.5-1.5B-Instruct as the reader, the attack succeeds on 96% of questions without a defence (prefix form). Filtering by set density lowers this to 8%, and filtering by retrieval score to 20%.

**Paraphrase Robustness.** Rewording a question does not change top-10 retrieval significantly (Hit@10 difference −0.003, p = 1.00). Paraphrases find a relevant passage within the top 100 more often (+10 percentage points, p = 0.0016).

## Repository structure

```
.
├── *.py                     experiment scripts (described below)
├── run_trust_pipeline.sh    runs the full trust-scoring pipeline
├── requirements_trust.txt   Python dependencies for the extensions
├── data_processed/          paraphrase inputs, embeddings and results
│   └── trust/               trust-scoring features and results
├── figures/                 result figures (PDF and PNG)
├── src/                     retriever, attack and model code from PoisonedRAG
├── model_configs/           LLM configurations for the original attack
└── results/                 original PoisonedRAG retrieval and attack outputs
```

## Installation

The extensions were tested with Python 3.14 on an Apple M2 (MPS); a CPU also works.

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements_trust.txt
.venv/bin/pip install sentence-transformers   # only for validate_paraphrases.py
```

All models are downloaded from Hugging Face on first use: `facebook/contriever`, `BeIR/query-gen-msmarco-t5-base-v1` (doc2query), GPT-2, `FacebookAI/roberta-large-mnli`, `deepset/roberta-base-squad2` and `Qwen/Qwen2.5-1.5B-Instruct`.

## Passage Trust Scoring

### Method

The defence scores each passage in the candidate pool without knowing the attack text. It uses 100 NQ target questions, a pool of 9,872 passages and PoisonedRAG's black-box poisoned passages, in two forms: with the question prefix and without it. The signals are:

| Signal | Idea |
|---|---|
| Relevance consistency (`z_drop`) | A poisoned passage should lose relevance when the question is paraphrased |
| Reverse-question focus | doc2query generates the questions a passage answers; a poisoned passage should answer only the target question |
| Retrieval score, set density, shortness | Properties of the retrieved set and the passage |
| GPT-2 perplexity | Fluency of the passage text |
| Reader support | Support score of an extractive reader for an answer in the passage |
| Rank-max combination | Each signal is scored as the fraction of clean passages it exceeds, and the largest is taken |

### Evaluation

- **Detection:** AUROC, AUPRC and TPR at 5% FPR, with 10,000 bootstrap resamples over questions, paired bootstrap tests and Holm correction.
- **Robustness:** the worst case over six settings: the two poisoned-passage forms, each with five and with one poisoned passage; a reworded attack, in which the attacker rewords the question prefix; and a diverse attack aimed at the reverse-question signal.
- **End-to-end effect:** attack success rate and accuracy with an extractive reader (`deepset/roberta-base-squad2`) and with Qwen2.5-1.5B-Instruct (retrieve 10, read 5).
- **Cost:** running time per question.

### Running the pipeline

`run_trust_pipeline.sh` runs every step in order from the files in `data_processed/`, and downloads the NQ qrels to `datasets/nq/qrels/test.tsv` if they are missing.

```bash
bash run_trust_pipeline.sh            # uses .venv/bin/python
bash run_trust_pipeline.sh python3    # or another interpreter
```

| Step | Scripts |
|---|---|
| 1. Evaluation set and features | `build_trust_eval_set.py`, `generate_trust_embeddings.py`, `generate_answer_support.py`, `generate_perplexity.py`, `compute_trust_scores.py` |
| 2. Detection | `evaluate_detection.py`, `evaluate_poison_count.py`, `analyse_failure_cases.py`, `analyse_why_consistency_fails.py` |
| 3. Paraphrase quality | `check_paraphrases_nli.py`, `evaluate_paraphrase_quality.py`, `compare_paraphrase_checks.py` |
| 4. End-to-end effect, reworded attack, combinations | `evaluate_end_to_end.py`, `evaluate_end_to_end_llm.py`, `evaluate_adaptive_attack.py`, `evaluate_combinations.py` |
| 5. Reverse questions | `generate_reverse_questions.py`, `generate_pool_focus.py`, `evaluate_reverse_questions.py`, `analyse_reverse_questions.py`, `evaluate_reverse_end_to_end.py`, `evaluate_diverse_attack.py`, `evaluate_robustness.py` |
| 6. Statistics and design-choice checks | `apply_holm_correction.py`, `evaluate_sensitivity.py` |
| 7. Running time and figures | `measure_cost.py`, `make_figures.py` |

Results are written to `data_processed/trust/` and figures to `figures/`.

> **Warning:** do not run `make_paraphrase_review_files.py`. It overwrites the manual paraphrase reviews. The pipeline does not call it.

### Reproducibility notes

- Contriever always runs on the CPU, because Apple MPS produced wrong embeddings for some batches. The other models use MPS when it is available.
- The trained combinations use `GroupKFold`, whose folds changed in scikit-learn 1.6. To reproduce the saved combination results exactly, install `scikit-learn<1.6` (1.5.2 was checked); newer versions shift some values slightly.
- `measure_cost.py` reports wall-clock time, so its results depend on the machine.

### Limitations

- One dataset (NQ), one retriever (Contriever) and a small LLM reader.
- Only the black-box attack is evaluated; the white-box HotFlip attack is not.

## Paraphrase Robustness

### Method

Each of the 100 NQ target questions has three paraphrases (400 queries in total). Contriever retrieval for the original questions is compared with retrieval for their paraphrases over the same 9,872-passage candidate pool, to answer three questions:

1. Do paraphrased queries find relevant passages more or less often than the original questions?
2. How consistent is retrieval across different wordings of the same question?
3. Are the differences statistically significant?

**Paraphrase validation.** Each paraphrase was compared with its original question by cosine similarity using `sentence-transformers/all-MiniLM-L6-v2`. Paraphrases flagged in a first pass (threshold 0.80) were corrected. In the corrected set used for the experiment, 280 of 300 paraphrases pass the 0.75 threshold; the 20 flagged paraphrases come from 14 questions. A flag marks a paraphrase for closer reading, not as wrong.

**Retrieval settings.**

| Parameter | Value |
|---|---|
| Retrieval model | `facebook/contriever` |
| Maximum sequence length | 512 tokens |
| Embedding dimension | 768 |
| Retrieval depth | Top 100 |
| Similarity | Dot product |

With these settings, the original-query results match the baseline in `data_processed/baseline_retrieval_metrics.json` on all seven metrics.

**Metrics.** Relevance comes from the NQ qrels.

- **Hit@k** (k = 1, 5, 10, 100): share of queries with at least one relevant passage in the top k.
- **Recall@k** (k = 10, 100): mean share of a query's relevant passages found in the top k.
- **MRR@10**: mean reciprocal rank of the first relevant passage in the top 10.

### Running the experiment

| Step | Script |
|---|---|
| 1. Generate paraphrases | `generate_reworded_questions.py` |
| 2. Score paraphrase similarity | `validate_paraphrases.py` |
| 3. Rebuild the candidate pool from the NQ corpus | `reconstruct_candidate_passages.py` |
| 4. Encode passages and queries | `generate_passage_embeddings.py`, `generate_query_embeddings.py` |
| 5. Retrieve for all 400 queries | `run_paraphrase_retrieval.py` |
| 6. Compute retrieval metrics | `evaluate_experiment_retrieval.py` |
| 7. Paired differences and significance tests | `evaluate_paraphrase_robustness.py`, `evaluate_paraphrase_significance.py` |

Step 3 needs the NQ corpus at `datasets/nq/corpus.jsonl` (see [Dataset](#dataset)). `validate_paraphrases.py` reads the generated paraphrases; the corrected set is in `data_processed/nq_reworded_questions_corrected.json` and its validation in `data_processed/corrected_paraphrase_validation.csv`.

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

Rewording mainly changes where relevant passages rank, rather than making retrieval better or worse overall. Because some paraphrases may not keep the exact meaning of the question, the Hit@100 gain should not be read as a general benefit of paraphrasing.

### Limitations

- 100 questions and 300 paraphrases limit statistical power.
- Embedding similarity does not guarantee that a paraphrase keeps the meaning, and not every paraphrase was checked by hand.
- Only retrieval is measured, not answer quality.
- Retrieval runs over the 9,872-passage candidate pool, not the full NQ corpus.
- The configuration that produced the baseline metrics was not independently confirmed.

## Original PoisonedRAG usage

![Illustration of PoisonedRAG](PoisonedRAG.png "Illustration of PoisonedRAG")

### Setup

```bash
conda create -n PoisonedRAG python=3.10
conda activate PoisonedRAG
pip install beir openai google-generativeai
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade charset-normalizer
pip3 install "fschat[model_worker,webui]"
```

### Dataset

Datasets are downloaded to `datasets/` on first use. To download them in advance:

```bash
python prepare_dataset.py
```

### API keys

To use PaLM 2, GPT-3.5, GPT-4 or LLaMA-2, add the API key to the matching file in `model_configs/`. For LLaMA-2 the key is a Hugging Face access token (request access on the [LLaMA-2 model page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)).

```json
"api_key_info":{
    "api_keys":[
        "Your api key here"
    ],
    "api_key_use": 0
},
```

### Reproducing the paper's results

`run.py` sets the LLM, dataset and attack settings; the defaults reproduce the paper's main results.

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

```bash
python run.py
```

To use another dataset, convert it to the BEIR format and compute retrieval scores with `evaluate_beir.py`.

## Acknowledgement

- The attack code is from [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) by Zou, Geng, Wang and Jia.
- PoisonedRAG builds on [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning), takes its model wrappers from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection), and uses the [BEIR](https://github.com/beir-cellar/beir) benchmark and [Contriever](https://github.com/facebookresearch/contriever) for retrieval.

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

## License

Released under the MIT License; see [LICENSE](LICENSE).
