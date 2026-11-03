"""Offline retrieval evaluation against future known-user positives."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from recommender.evaluation.metrics import ndcg_at_k, recall_at_k
from recommender.retrieval.data import EvaluationUser
from recommender.retrieval.model import IdEmbeddingTower, embed_ids


def _top_unseen_items(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_index: dict[int, int],
    seen_items: frozenset[int],
    k: int,
) -> list[int]:
    scores = scores.copy()
    seen_indices = [
        candidate_index[movie_id]
        for movie_id in seen_items
        if movie_id in candidate_index
    ]
    if seen_indices:
        scores[np.asarray(seen_indices, dtype=np.int64)] = -np.inf

    available = min(k, len(candidate_ids) - len(seen_indices))
    if available <= 0:
        return []
    indices = np.argpartition(scores, -available)[-available:]
    ordered = indices[np.argsort(scores[indices])[::-1]]
    return [int(movie_id) for movie_id in candidate_ids[ordered]]


def _popular_unseen_items(
    popularity_order: list[int], seen_items: frozenset[int], k: int
) -> list[int]:
    recommendations = []
    for movie_id in popularity_order:
        if movie_id not in seen_items:
            recommendations.append(movie_id)
            if len(recommendations) >= k:
                break
    return recommendations


def evaluate_retrieval(
    user_tower: IdEmbeddingTower,
    item_tower: IdEmbeddingTower,
    candidate_ids: np.ndarray,
    evaluation_users: list[EvaluationUser],
    popularity_order: list[int],
    evaluation_ks: tuple[int, ...],
    embedding_batch_size: int,
    user_batch_size: int = 64,
) -> tuple[dict, np.ndarray]:
    """Compare learned exact retrieval with popularity on identical users."""
    if not evaluation_users:
        raise ValueError("At least one evaluation user is required")
    max_k = max(evaluation_ks)
    candidate_index = {
        int(movie_id): index for index, movie_id in enumerate(candidate_ids)
    }
    item_embeddings = embed_ids(item_tower, candidate_ids, embedding_batch_size)
    metric_values: dict[str, dict[str, list[float]]] = {
        "two_tower": defaultdict(list),
        "popularity": defaultdict(list),
    }

    started = time.perf_counter()
    for start in range(0, len(evaluation_users), user_batch_size):
        users = evaluation_users[start : start + user_batch_size]
        user_ids = np.asarray([user.user_id for user in users], dtype=np.int64)
        user_embeddings = embed_ids(user_tower, user_ids, user_batch_size)
        score_matrix = user_embeddings @ item_embeddings.T

        for row, user in enumerate(users):
            model_recommendations = _top_unseen_items(
                score_matrix[row],
                candidate_ids,
                candidate_index,
                user.seen_items,
                max_k,
            )
            popularity_recommendations = _popular_unseen_items(
                popularity_order, user.seen_items, max_k
            )
            for k in evaluation_ks:
                metric_values["two_tower"][f"recall@{k}"].append(
                    recall_at_k(model_recommendations, user.relevant_items, k)
                )
                metric_values["two_tower"][f"ndcg@{k}"].append(
                    ndcg_at_k(model_recommendations, user.relevant_items, k)
                )
                metric_values["popularity"][f"recall@{k}"].append(
                    recall_at_k(popularity_recommendations, user.relevant_items, k)
                )
                metric_values["popularity"][f"ndcg@{k}"].append(
                    ndcg_at_k(popularity_recommendations, user.relevant_items, k)
                )

    elapsed = time.perf_counter() - started
    averaged = {
        method: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for method, metrics in metric_values.items()
    }
    report = {
        "evaluated_users": len(evaluation_users),
        "candidate_items": len(candidate_ids),
        "future_relevant_items": sum(
            len(user.relevant_items) for user in evaluation_users
        ),
        "seen_items_filtered": True,
        "evaluation_seconds": elapsed,
        "users_per_second": len(evaluation_users) / elapsed,
        "metrics": averaged,
    }
    return report, item_embeddings
