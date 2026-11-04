"""Candidate-level and listwise evaluation for ranking models."""

from __future__ import annotations

import math

import numpy as np


def _list_metrics(
    labels: np.ndarray, total_relevant: int, k: int
) -> tuple[float, float]:
    hits = float(labels[:k].sum())
    recall = hits / total_relevant if total_relevant else 0.0
    dcg = sum(
        float(label) / math.log2(rank + 2)
        for rank, label in enumerate(labels[:k])
    )
    ideal_hits = min(k, total_relevant)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return recall, dcg / idcg if idcg else 0.0


def evaluate_candidate_order(
    user_ids: np.ndarray,
    labels: np.ndarray,
    total_relevant_counts: np.ndarray,
    ordering_scores: np.ndarray,
    ks: tuple[int, ...],
    descending: bool = True,
) -> dict[str, float | int]:
    """Macro-average end-to-end Recall/NDCG over one candidate list per user."""
    arrays = [
        np.asarray(user_ids),
        np.asarray(labels),
        np.asarray(total_relevant_counts),
        np.asarray(ordering_scores),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("ranking evaluation inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("ranking evaluation inputs must have equal length")
    results = {f"recall_at_{k}": [] for k in ks}
    results.update({f"ndcg_at_{k}": [] for k in ks})
    for user_id in np.unique(arrays[0]):
        positions = np.flatnonzero(arrays[0] == user_id)
        direction = -arrays[3][positions] if descending else arrays[3][positions]
        order = np.argsort(direction, kind="stable")
        ordered_labels = arrays[1][positions][order]
        total_relevant = int(arrays[2][positions][0])
        for k in ks:
            recall, ndcg = _list_metrics(ordered_labels, total_relevant, k)
            results[f"recall_at_{k}"].append(recall)
            results[f"ndcg_at_{k}"].append(ndcg)
    return {
        "users": len(np.unique(arrays[0])),
        **{name: float(np.mean(values)) for name, values in results.items()},
    }
