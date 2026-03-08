"""
Evaluation metrics for video captioning.

BLEU, ROUGE-L, and Recall@K for retrieval evaluation.
"""

import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    """Simple tokenization for BLEU/ROUGE."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute BLEU score (sentence-level, smoothed).

    Args:
        reference: Ground truth caption
        hypothesis: Model prediction
        max_n: Max n-gram order

    Returns:
        BLEU score in [0, 1]
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(zip(*(ref_tokens[i:] for i in range(n))))
        hyp_ngrams = Counter(zip(*(hyp_tokens[i:] for i in range(n))))
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0.0)

    # Smoothed: avoid log(0)
    log_prec = sum(np.log(p + 1e-10) for p in precisions) / max_n
    brevity = len(ref_tokens) / len(hyp_tokens) if hyp_tokens else 0
    bp = min(1.0, brevity) if brevity > 0 else 0.0
    return bp * np.exp(log_prec)


def rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L F1 (longest common subsequence).

    Args:
        reference: Ground truth
        hypothesis: Prediction

    Returns:
        ROUGE-L F1 in [0, 1]
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: list,
    gallery_labels: list,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Recall@K for retrieval: given query, retrieve from gallery.

    Args:
        query_embeddings: [N_q, D]
        gallery_embeddings: [N_g, D]
        query_labels: label per query (e.g. video_id)
        gallery_labels: label per gallery item
        k_values: K values to compute

    Returns:
        {"R@1": float, "R@5": float, "R@10": float}
    """
    # L2 normalize
    q = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    g = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(q, g.T)  # [N_q, N_g]

    results = {}
    for k in k_values:
        topk = np.argsort(-sim, axis=1)[:, :k]
        hits = 0
        for i, qlabel in enumerate(query_labels):
            retrieved = [gallery_labels[j] for j in topk[i]]
            if qlabel in retrieved:
                hits += 1
        results[f"R@{k}"] = hits / len(query_labels) if query_labels else 0.0
    return results


def compute_caption_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """
    Aggregate BLEU and ROUGE-L over a list of caption pairs.

    Args:
        references: Ground truth captions
        hypotheses: Model predictions

    Returns:
        {"BLEU": float, "ROUGE-L": float}
    """
    assert len(references) == len(hypotheses)
    bleu_scores = [bleu(r, h) for r, h in zip(references, hypotheses)]
    rouge_scores = [rouge_l(r, h) for r, h in zip(references, hypotheses)]
    return {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-L": np.mean(rouge_scores),
    }
