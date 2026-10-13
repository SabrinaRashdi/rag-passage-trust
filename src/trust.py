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
