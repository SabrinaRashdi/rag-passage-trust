import re
import string
from collections import Counter

import numpy as np

# Number of top clean passages used to describe how a query "normally" scores.
REFERENCE_TOP_N = 100


def reference_stats(clean_scores, top_n=REFERENCE_TOP_N):
    """Mean and std of the top-n clean pool scores for one query variant."""
    top = np.sort(clean_scores)[::-1][:top_n]
    return float(top.mean()), float(top.std())


def z_score(score, clean_scores, top_n=REFERENCE_TOP_N):
    """Passage score relative to what this query wording normally retrieves."""
    mean, std = reference_stats(clean_scores, top_n)
    return (score - mean) / std


def pool_rank(score, clean_scores, exclude_index=None):
    """Rank the passage would get in the clean pool (1 = most relevant)."""
    higher = clean_scores > score
    if exclude_index is not None:
        higher[exclude_index] = False
    return int(higher.sum()) + 1


def consistency_drop(original, paraphrases):
    """
    Relevance-consistency signal: how much relevance is lost when the query
    is paraphrased. Large positive values mean the passage is relevant mainly
    for the original wording, which is what we treat as suspicious.

    original: array (n,), paraphrases: array (n, k)
    """
    paraphrases = np.asarray(paraphrases, dtype=float)
    return np.asarray(original, dtype=float) - paraphrases.mean(axis=1)


def set_density(embeddings, neighbours):
    """
    Set-level signal: mean cosine similarity of each passage to its nearest
    neighbours inside one candidate set. Poisoned passages written for the
    same question tend to sit close together.

    embeddings: array (n, d), returns array (n,)
    """
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    return np.sort(similarity, axis=1)[:, -neighbours:].mean(axis=1)


def normalize_answer(text):
    """SQuAD-style answer normalisation: lower case, no punctuation/articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(answer_a, answer_b):
    """Token-level F1 overlap between two extracted answers (0 to 1)."""
    tokens_a = normalize_answer(answer_a).split()
    tokens_b = normalize_answer(answer_b).split()

    if not tokens_a or not tokens_b:
        return float(tokens_a == tokens_b)

    common = sum((Counter(tokens_a) & Counter(tokens_b)).values())

    if common == 0:
        return 0.0

    precision = common / len(tokens_a)
    recall = common / len(tokens_b)
    return 2 * precision * recall / (precision + recall)
