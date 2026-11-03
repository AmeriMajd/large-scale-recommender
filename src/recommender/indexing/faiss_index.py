"""Small, testable FAISS primitives used by the Day 4 pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np


def validate_index_inputs(
    movie_ids: np.ndarray, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the retrieval export and return FAISS-compatible arrays."""
    ids = np.asarray(movie_ids)
    vectors = np.asarray(embeddings)
    if ids.ndim != 1:
        raise ValueError("movie_ids must be one-dimensional")
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if len(ids) != len(vectors):
        raise ValueError("movie_ids and embeddings must have the same row count")
    if len(ids) == 0:
        raise ValueError("cannot build an index from zero vectors")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("movie_ids must be unique")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("movie_ids must contain integers")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("embeddings contain NaN or infinity")

    ids = np.ascontiguousarray(ids, dtype=np.int64)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must not contain zero vectors")
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embeddings must be L2-normalized for cosine search")
    return ids, vectors


def build_exact_index(movie_ids: np.ndarray, embeddings: np.ndarray) -> faiss.Index:
    """Build an exhaustive inner-product index with external movie IDs."""
    ids, vectors = validate_index_inputs(movie_ids, embeddings)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)
    return index


def build_hnsw_index(
    movie_ids: np.ndarray,
    embeddings: np.ndarray,
    m: int,
    ef_construction: int,
) -> faiss.Index:
    """Build an HNSW inner-product index with external movie IDs."""
    if m <= 0 or ef_construction <= 0:
        raise ValueError("HNSW construction parameters must be positive")
    ids, vectors = validate_index_inputs(movie_ids, embeddings)
    storage = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    storage.hnsw.efConstruction = ef_construction
    index = faiss.IndexIDMap2(storage)
    index.add_with_ids(vectors, ids)
    return index


def set_hnsw_ef_search(index: faiss.Index, ef_search: int) -> None:
    """Set HNSW search depth through wrappers such as IndexIDMap2."""
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", ef_search)


def write_and_reload(index: faiss.Index, path: Path) -> faiss.Index:
    """Persist an index and immediately reload it for verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return faiss.read_index(str(path))


def approximate_recall_at_k(
    exact_ids: np.ndarray, approximate_ids: np.ndarray
) -> float:
    """Average set overlap with exact top-K results."""
    exact = np.asarray(exact_ids)
    approximate = np.asarray(approximate_ids)
    if exact.ndim != 2 or approximate.ndim != 2:
        raise ValueError("search result arrays must be two-dimensional")
    if exact.shape != approximate.shape:
        raise ValueError("exact and approximate results must have equal shapes")
    if exact.shape[1] == 0:
        raise ValueError("K must be positive")
    recalls = [
        len(set(expected).intersection(predicted)) / exact.shape[1]
        for expected, predicted in zip(exact, approximate, strict=True)
    ]
    return float(np.mean(recalls))


def _latency_summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def benchmark_index(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    warmup_queries: int,
    repeats: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Measure client-visible single-query latency and batched throughput."""
    query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
    if query_matrix.ndim != 2 or len(query_matrix) == 0:
        raise ValueError("queries must be a non-empty two-dimensional matrix")
    if k <= 0 or k > index.ntotal:
        raise ValueError("k must be positive and no larger than the index")
    if repeats <= 0 or warmup_queries < 0:
        raise ValueError("invalid benchmark repeat or warmup count")

    if warmup_queries:
        index.search(query_matrix[: min(warmup_queries, len(query_matrix))], k)

    latencies: list[float] = []
    for _ in range(repeats):
        for query in query_matrix:
            started = time.perf_counter_ns()
            index.search(query.reshape(1, -1), k)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)

    batch_seconds: list[float] = []
    result_ids = np.empty((0, k), dtype=np.int64)
    for _ in range(repeats):
        started = time.perf_counter()
        _, result_ids = index.search(query_matrix, k)
        batch_seconds.append(time.perf_counter() - started)
    best_batch_seconds = min(batch_seconds)
    metrics: dict[str, float | int] = {
        **_latency_summary(latencies),
        "batch_queries_per_second": float(len(query_matrix) / best_batch_seconds),
        "batch_best_seconds": float(best_batch_seconds),
        "query_count": len(query_matrix),
        "repeats": repeats,
    }
    return metrics, result_ids
