import numpy as np
import pytest

from recommender.ranking.evaluate import evaluate_candidate_order


def test_candidate_metrics_include_unretrieved_relevant_items_in_denominator() -> None:
    metrics = evaluate_candidate_order(
        user_ids=np.asarray([1, 1, 1, 2, 2, 2]),
        labels=np.asarray([1, 0, 0, 0, 1, 0]),
        total_relevant_counts=np.asarray([2, 2, 2, 1, 1, 1]),
        ordering_scores=np.asarray([3, 2, 1, 3, 2, 1]),
        ks=(1,),
    )

    assert metrics["recall_at_1"] == pytest.approx(0.25)
    assert metrics["ndcg_at_1"] == pytest.approx(0.5)


def test_candidate_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        evaluate_candidate_order(
            np.asarray([1]),
            np.asarray([1, 0]),
            np.asarray([1]),
            np.asarray([1]),
            (1,),
        )
