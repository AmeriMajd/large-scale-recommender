"""Small, framework-independent ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(
    recommendations: Sequence[int], relevant_items: Collection[int], k: int
) -> float:
    """Return the fraction of relevant items found in the first ``k`` results."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    return len(set(recommendations[:k]) & relevant) / len(relevant)


def ndcg_at_k(
    recommendations: Sequence[int], relevant_items: Collection[int], k: int
) -> float:
    """Return binary Normalized Discounted Cumulative Gain at ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(recommendations[:k])
        if item_id in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg
