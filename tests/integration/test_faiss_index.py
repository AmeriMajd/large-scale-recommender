from pathlib import Path

import numpy as np
import pytest

from recommender.indexing.faiss_index import (
    approximate_recall_at_k,
    build_exact_index,
    build_hnsw_index,
    set_hnsw_ef_search,
    validate_index_inputs,
    write_and_reload,
)


def _normalized_random_vectors(count: int, dimension: int) -> np.ndarray:
    vectors = np.random.default_rng(42).normal(size=(count, dimension)).astype(
        np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def test_exact_index_returns_external_ids_and_survives_reload(tmp_path: Path) -> None:
    movie_ids = np.asarray([101, 305, 999], dtype=np.int64)
    vectors = np.eye(3, dtype=np.float32)
    index = build_exact_index(movie_ids, vectors)

    distances, result_ids = index.search(vectors[[1]], 3)
    assert result_ids[0, 0] == 305
    assert distances[0, 0] == pytest.approx(1.0)

    reloaded = write_and_reload(index, tmp_path / "exact.faiss")
    loaded_distances, loaded_ids = reloaded.search(vectors, 3)
    original_distances, original_ids = index.search(vectors, 3)
    np.testing.assert_array_equal(loaded_ids, original_ids)
    np.testing.assert_allclose(loaded_distances, original_distances)


def test_exact_index_matches_numpy_inner_product_top_k() -> None:
    vectors = _normalized_random_vectors(100, 8)
    movie_ids = np.arange(5_000, 5_100, dtype=np.int64)
    queries = _normalized_random_vectors(7, 8)
    index = build_exact_index(movie_ids, vectors)

    distances, result_ids = index.search(queries, 10)
    scores = queries @ vectors.T
    expected_rows = np.argsort(-scores, axis=1)[:, :10]

    np.testing.assert_array_equal(result_ids, movie_ids[expected_rows])
    np.testing.assert_allclose(
        distances, np.take_along_axis(scores, expected_rows, axis=1), atol=1e-6
    )


def test_hnsw_has_high_overlap_with_exact_search() -> None:
    vectors = _normalized_random_vectors(500, 16)
    movie_ids = np.arange(10_000, 10_500, dtype=np.int64)
    queries = vectors[::10]
    exact = build_exact_index(movie_ids, vectors)
    hnsw = build_hnsw_index(movie_ids, vectors, m=16, ef_construction=100)
    set_hnsw_ef_search(hnsw, 128)

    _, exact_ids = exact.search(queries, 20)
    _, hnsw_ids = hnsw.search(queries, 20)
    assert approximate_recall_at_k(exact_ids, hnsw_ids) >= 0.95


@pytest.mark.parametrize(
    ("movie_ids", "vectors", "message"),
    [
        (np.asarray([1, 1]), np.eye(2, dtype=np.float32), "unique"),
        (np.asarray([1]), np.zeros((1, 2), dtype=np.float32), "zero"),
        (np.asarray([1]), np.asarray([[np.nan, 0.0]]), "NaN"),
    ],
)
def test_invalid_index_inputs_are_rejected(
    movie_ids: np.ndarray, vectors: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_index_inputs(movie_ids, vectors)
