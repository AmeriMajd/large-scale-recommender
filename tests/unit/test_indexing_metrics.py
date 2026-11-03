import numpy as np
import pytest

from recommender.indexing.faiss_index import approximate_recall_at_k


def test_approximate_recall_uses_set_overlap_per_query() -> None:
    exact = np.asarray([[1, 2, 3], [4, 5, 6]])
    approximate = np.asarray([[3, 2, 9], [8, 9, 4]])

    assert approximate_recall_at_k(exact, approximate) == pytest.approx(0.5)


def test_approximate_recall_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        approximate_recall_at_k(np.ones((2, 3)), np.ones((2, 2)))
