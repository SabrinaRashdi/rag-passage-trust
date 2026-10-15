
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForCausalLM, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, AutoTokenizer,
)

from compute_trust_scores import (
    SET_NEIGHBOURS, VARIANTS, load_embeddings, reference_masks,
)
from evaluate_detection import N_FOLDS, auroc, tpr_at_fpr
from evaluate_end_to_end import (
    COMBINED, COMBINED_FEATURES, PASSAGE_METHODS, READ_K, RETRIEVE_K,
    answer_matches,
)
from generate_answer_support import MODEL_NAME as READER_NAME
from generate_answer_support import BATCH_SIZE, read
from generate_perplexity import MAX_LENGTH as PERPLEXITY_MAX_LENGTH
from generate_perplexity import MODEL_NAME as PERPLEXITY_NAME
from generate_trust_embeddings import MODEL_NAME as RETRIEVER_NAME
from generate_trust_embeddings import encode
from measure_cost import PARAPHRASER_NAME, release
from src.contriever_src.contriever import Contriever
from src.trust import (
    answer_f1, consistency_drop, normalize_answer, set_density, z_score,
)

# Adaptive attack: the attacker knows that copying the question is easy to
# spot, so each of the 5 poisoned passages starts with a different rewording
# of the question instead of the question itself. The defences are kept
# exactly as tuned for the normal attack (Kind 1): same thresholds, and the
# combined score is trained on Kind 1 passages only.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"

DATASET_PATH = DATA_DIR / "nq_passage_dataset.json"
QUERIES_PATH = DATA_DIR / "experiment_queries.json"
EVAL_PATH = TRUST_DIR / "eval_passages.jsonl"
SCORES_PATH = TRUST_DIR / "trust_scores.csv"
ANSWER_SUPPORT_PATH = TRUST_DIR / "answer_support.csv"
END_TO_END_PATH = TRUST_DIR / "end_to_end_results.json"

PASSAGES_PATH = TRUST_DIR / "adaptive_attack_passages.jsonl"
OUTPUT_PATH = TRUST_DIR / "adaptive_attack_results.json"
# Features of the candidate sets under this attack, for evaluate_combinations.py
SCORES_PATH_OUT = TRUST_DIR / "adaptive_attack_scores.csv"
SAVED_FEATURES = (
    "score_original", "support_original", *COMBINED_FEATURES,
    "log_perplexity", "num_words",
)

KNOWN_ATTACK = "prefix"
ATTACK = "reworded"
N_POISONED = 5
# The attacker uses more beam candidates than the defence (which kept the
# first 3), and never uses a wording that the defence uses.
N_CANDIDATES = 10


def attacker_rewordings(questions, device):
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_NAME)
    model = model.to(device)
    model.eval()

    rewordings = {}

    for qid, (question, defence_wordings) in questions.items():
        inputs = tokenizer(
            "paraphrase: " + question,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=N_CANDIDATES,
                num_return_sequences=N_CANDIDATES,
                do_sample=False,
                early_stopping=True,
            )

        seen = {normalize_answer(w) for w in defence_wordings}
        chosen = []

        for candidate in tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        ):
            key = normalize_answer(candidate)

            if key and key not in seen:
                chosen.append(candidate.strip().rstrip("?.!"))
                seen.add(key)

        assert chosen, f"no new rewording for {qid}"
        # Reuse rewordings if fewer than 5 new ones were found.
        rewordings[qid] = [chosen[i % len(chosen)] for i in range(N_POISONED)]

    release(tokenizer, model)
    return rewordings


def passage_features(passages, new_emb, attack, queries, query_emb,
                     query_index, pool_emb, references, device):
    """Reader answers and support, GPT-2 perplexity and the consistency
    features of compute_trust_scores.py for new poisoned passages
    (dicts with passage_id, query_id and text). Returns the rows and the
    reader outputs."""
    texts = [p["text"] for p in passages]

    # Reader answers and support for every question wording.
    tokenizer = AutoTokenizer.from_pretrained(READER_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(READER_NAME)
    model = model.to(device)
    model.eval()

    reader = {p["passage_id"]: {} for p in passages}

    for variant in VARIANTS:
        for start in range(0, len(passages), BATCH_SIZE):
            batch = passages[start:start + BATCH_SIZE]
            outputs = read(
                [queries[f"{p['query_id']}_{variant}"] for p in batch],
                [p["text"] for p in batch],
                tokenizer, model, device,
            )

            for p, out in zip(batch, outputs):
                reader[p["passage_id"]][f"answer_{variant}"] = out["answer"]
                reader[p["passage_id"]][f"support_{variant}"] = out["support"]

    release(tokenizer, model)

    # GPT-2 perplexity.
    tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_NAME)
    model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_NAME).to(device)
    model.eval()
    perplexity = []

    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(
                text, truncation=True, max_length=PERPLEXITY_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            perplexity.append(
                float(model(**inputs, labels=inputs["input_ids"]).loss)
            )

    release(tokenizer, model)

    # The same features as compute_trust_scores.py.
    new_rows = []

    for p, embedding, log_ppl in zip(passages, new_emb, perplexity):
        qid = p["query_id"]
        variants = np.stack([
            query_emb[query_index[f"{qid}_{v}"]] for v in VARIANTS
        ])
        clean_scores = variants @ pool_emb[references[qid]].T
        scores = variants @ embedding
        answers = reader[p["passage_id"]]

        row = {
            "passage_id": p["passage_id"], "query_id": qid, "label": 1,
            "attack": attack, "is_gold": 0,
            "log_perplexity": log_ppl, "num_words": len(p["text"].split()),
        }

        for v, name in enumerate(VARIANTS):
            row[f"score_{name}"] = float(scores[v])
            row[f"z_{name}"] = z_score(scores[v], clean_scores[v])
            row[f"support_{name}"] = float(answers[f"support_{name}"])

        z_paras = [row[f"z_{n}"] for n in VARIANTS[1:]]
        support_paras = [row[f"support_{n}"] for n in VARIANTS[1:]]
        row["z_drop"] = float(consistency_drop(
            [row["z_original"]], [z_paras]
        )[0])
        row["support_drop"] = float(consistency_drop(
            [row["support_original"]], [support_paras]
        )[0])
        row["answer_disagreement"] = 1 - float(np.mean([
            answer_f1(answers["answer_original"], answers[f"answer_{n}"])
            for n in VARIANTS[1:]
        ]))
        new_rows.append(row)

    return new_rows, reader


def trained_on_known_attack(known_features, known_labels, known_groups,
                            test_features, test_groups):
    """Logistic regression trained on the known attack (Kind 1). Each test
    passage is scored by a model that never saw its question."""
    scores = np.zeros(len(test_features))

    for train, held_out in GroupKFold(n_splits=N_FOLDS).split(
        known_features, known_labels, known_groups
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(known_features[train], known_labels[train])
        target = np.isin(test_groups, np.unique(known_groups[held_out]))
        scores[target] = model.decision_function(test_features[target])

    return scores


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)

    with EVAL_PATH.open(encoding="utf-8") as f:
        eval_passages = {
            p["passage_id"]: p for p in (json.loads(line) for line in f)
        }

    with SCORES_PATH.open(encoding="utf-8") as f:
        known_rows = list(csv.DictReader(f))

    with ANSWER_SUPPORT_PATH.open(encoding="utf-8") as f:
        known_reader = {row["passage_id"]: row for row in csv.DictReader(f)}

    with END_TO_END_PATH.open(encoding="utf-8") as f:
        thresholds = {
            method: values["threshold"]
            for method, values in json.load(f)[KNOWN_ATTACK].items()
            if values["threshold"] is not None
        }

    questions = sorted(dataset)

    # 1. Poisoned passages with a reworded question in front.
    rewordings = attacker_rewordings(
        {
            qid: (dataset[qid]["question"],
                  [queries[f"{qid}_{v}"] for v in VARIANTS])
            for qid in questions
        },
        device,
    )

    passages = []

    for qid in questions:
        for j, (prefix, text) in enumerate(zip(
            rewordings[qid], dataset[qid]["adversarial_passages"]
        ), 1):
            passages.append({
                "passage_id": f"{qid}::adv{j}_{ATTACK}",
                "query_id": qid,
                "prefix": prefix,
                # Same format as the PoisonedRAG attack: prefix + "." + text.
                "text": prefix + "." + text,
            })

    with PASSAGES_PATH.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    texts = [p["text"] for p in passages]

    # 2. Embeddings of the new passages and of the question wordings.
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_NAME)
    model = Contriever.from_pretrained(RETRIEVER_NAME).to(device)
    model.eval()
    new_emb = encode(texts, tokenizer, model, device, ATTACK)
    release(tokenizer, model)

    pool_emb, pool_index = load_embeddings("pool")
    query_emb, query_index = load_embeddings("query")
    adv_emb, adv_index = load_embeddings("poisoned")
    references = reference_masks(pool_index, eval_passages.values())

    # 3.-5. Reader, perplexity and the features of compute_trust_scores.py.
    new_rows, reader = passage_features(
        passages, new_emb, ATTACK, queries, query_emb, query_index,
        pool_emb, references, device,
    )

    new_reader = {
        pid: {k: str(v) for k, v in values.items()}
        for pid, values in reader.items()
    }
    benign = [r for r in known_rows if r["attack"] == "none"]
    known = [r for r in known_rows if r["attack"] in ("none", KNOWN_ATTACK)]

    def column(rows, name):
        return np.array([float(r[name]) for r in rows])

    # Combined score: trained on Kind 1 only (the attack the defender knows),
    # always on other questions than the ones it scores.
    test_rows = benign + new_rows
    test_groups = np.array([r["query_id"] for r in test_rows])
    combined = trained_on_known_attack(
        np.stack([column(known, n) for n in COMBINED_FEATURES], axis=1),
        np.array([int(r["label"]) for r in known]),
        np.array([r["query_id"] for r in known]),
        np.stack([column(test_rows, n) for n in COMBINED_FEATURES], axis=1),
        test_groups,
    )

    # Set density inside each question's candidate set (benign + poisoned).
    def embedding_of(row):
        if row["attack"] == ATTACK:
            return new_emb[new_rows.index(row)]
        if int(row["label"]) == 1:
            return adv_emb[adv_index[row["passage_id"]]]
        doc_id = eval_passages[row["passage_id"]]["doc_id"]
        return pool_emb[pool_index[doc_id]]

    test_embeddings = np.stack([embedding_of(r) for r in test_rows])
    density = np.zeros(len(test_rows))

    for qid in questions:
        members = np.flatnonzero(test_groups == qid)
        density[members] = set_density(
            test_embeddings[members], SET_NEIGHBOURS
        )

    labels = np.array([int(r["label"]) for r in test_rows])
    method_scores = {
        name: sign * column(test_rows, col)
        for name, (col, sign) in PASSAGE_METHODS.items()
    }
    method_scores[COMBINED] = combined
    method_scores["set_density"] = density

    with SCORES_PATH_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "passage_id", "query_id", "label", *SAVED_FEATURES, "set_density",
        ])

        for row, value in zip(test_rows, density):
            writer.writerow([
                row["passage_id"], row["query_id"], row["label"],
                *[row[name] for name in SAVED_FEATURES], float(value),
            ])

    with (TRUST_DIR / "detection_results.json").open(encoding="utf-8") as f:
        known_detection = json.load(f)[f"{KNOWN_ATTACK}/all_benign"]["methods"]

    results = {"detection": {}, "end_to_end": {}}

    for name, scores in method_scores.items():
        results["detection"][name] = {
            "auroc": auroc(labels, scores),
            "tpr_at_5pct_fpr": tpr_at_fpr(labels, scores),
            "auroc_known_attack": known_detection[name]["auroc"],
        }

    # 6. End to end: retrieve top 10, remove flagged passages, read top 5.
    row_index = {r["passage_id"]: i for i, r in enumerate(test_rows)}
    answers = {**known_reader, **new_reader}

    candidates = {}

    for qid in questions:
        pool = [
            r["passage_id"] for r in test_rows
            if r["query_id"] == qid and (
                r["attack"] == ATTACK
                or eval_passages[r["passage_id"]]["retrieved_top10"]
            )
        ]
        pool.sort(key=lambda pid: -float(
            test_rows[row_index[pid]]["score_original"]
        ))
        candidates[qid] = pool[:RETRIEVE_K]

    list_density = {}

    for qid, pids in candidates.items():
        values = set_density(
            test_embeddings[[row_index[pid] for pid in pids]], SET_NEIGHBOURS
        )
        list_density.update(zip(pids, values))

    for method in ("no_defence", *method_scores):
        success, correct, poison_read = [], [], []

        for qid in questions:
            pids = candidates[qid]

            if method == "no_defence":
                kept = pids
            elif method == "set_density":
                kept = [p for p in pids
                        if list_density[p] <= thresholds[method]]
            else:
                kept = [p for p in pids
                        if method_scores[method][row_index[p]]
                        <= thresholds[method]]

            read_ids = kept[:READ_K]
            answer = ""

            if read_ids:
                best = max(read_ids, key=lambda p: float(
                    answers[p]["support_original"]
                ))
                answer = answers[best]["answer_original"]

            success.append(
                answer_matches(answer, dataset[qid]["incorrect_answer"])
            )
            correct.append(
                answer_matches(answer, dataset[qid]["correct_answer"])
            )
            poison_read.append(sum(
                test_rows[row_index[p]]["attack"] == ATTACK for p in read_ids
            ))

        results["end_to_end"][method] = {
            "attack_success": float(np.mean(success)),
            "correct": float(np.mean(correct)),
            "mean_poison_read": float(np.mean(poison_read)),
        }

    with END_TO_END_PATH.open(encoding="utf-8") as f:
        known_end_to_end = json.load(f)[KNOWN_ATTACK]

    for method, values in results["end_to_end"].items():
        values["attack_success_known_attack"] = (
            known_end_to_end[method]["attacked"]["attack_success"]["rate"]
        )

    results["setup"] = {
        "questions": len(questions),
        "poisoned_passages": len(new_rows),
        "questions_with_5_different_rewordings": sum(
            len(set(rewordings[q])) == N_POISONED for q in questions
        ),
        "thresholds_from": f"{END_TO_END_PATH.name} ({KNOWN_ATTACK})",
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nADAPTIVE ATTACK: REWORDED QUESTION IN FRONT OF EACH PASSAGE")
    print("(defences tuned on the normal attack, Kind 1)")
    print("=" * 76)
    print("Questions with 5 different rewordings:",
          results["setup"]["questions_with_5_different_rewordings"])

    print(f"\n{'method':<22}{'AUROC Kind 1':>14}{'AUROC reworded':>16}"
          f"{'TPR@5%FPR':>12}")

    for name, r in results["detection"].items():
        print(f"{name:<22}{r['auroc_known_attack']:>14.3f}"
              f"{r['auroc']:>16.3f}{r['tpr_at_5pct_fpr']:>12.3f}")

    print(f"\n{'method':<22}{'success Kind 1':>16}{'success reworded':>18}"
          f"{'correct':>10}{'poison read':>13}")

    for name, r in results["end_to_end"].items():
        print(f"{name:<22}{r['attack_success_known_attack']:>16.2f}"
              f"{r['attack_success']:>18.2f}{r['correct']:>10.2f}"
              f"{r['mean_poison_read']:>13.2f}")

    print("\nPassages saved to:", PASSAGES_PATH)
    print("Features saved to:", SCORES_PATH_OUT)
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
