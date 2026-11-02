import pytest

from recommender.evaluation.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_counts_future_relevant_items() -> None:
    assert recall_at_k([5, 2, 9], {2, 8}, 3) == 0.5


def test_ndcg_rewards_relevant_items_near_the_top() -> None:
    first = ndcg_at_k([2, 5, 9], {2}, 3)
    third = ndcg_at_k([5, 9, 2], {2}, 3)

    assert first == 1.0
    assert third < first


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_metrics_reject_non_positive_k(metric) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        metric([1], {1}, 0)
