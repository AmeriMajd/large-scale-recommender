"""Build persistent exact and HNSW FAISS indexes from Day 3 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from recommender.config import ProjectConfig, load_config  # noqa: E402
from recommender.indexing.faiss_index import (  # noqa: E402
    approximate_recall_at_k,
    benchmark_index,
    build_exact_index,
    build_hnsw_index,
    set_hnsw_ef_search,
    validate_index_inputs,
    write_and_reload,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_retrieval_run(model_dir: Path) -> tuple[str, Path]:
    latest_path = model_dir / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"retrieval pointer does not exist: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(latest["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"retrieval run directory does not exist: {run_dir}")
    return str(latest["run_id"]), run_dir


def _make_user_queries(
    run_dir: Path, query_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    user_ids = np.load(run_dir / "user_ids.npy", allow_pickle=False)
    if user_ids.ndim != 1 or len(user_ids) == 0:
        raise ValueError("exported user_ids.npy must be a non-empty vector")
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        user_ids, size=min(query_count, len(user_ids)), replace=False
    ).astype(np.int64)
    signature = tf.saved_model.load(str(run_dir / "user_tower")).signatures[
        "serving_default"
    ]
    embeddings = signature(user_id=tf.constant(selected))["embedding"].numpy()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise ValueError("user tower returned non-normalized query embeddings")
    return selected, embeddings


def _verify_reload(
    original: faiss.Index,
    reloaded: faiss.Index,
    queries: np.ndarray,
    k: int,
) -> None:
    if original.ntotal != reloaded.ntotal:
        raise ValueError("reloaded index has a different vector count")
    check_queries = queries[: min(10, len(queries))]
    original_distances, original_ids = original.search(check_queries, k)
    loaded_distances, loaded_ids = reloaded.search(check_queries, k)
    np.testing.assert_array_equal(loaded_ids, original_ids)
    np.testing.assert_allclose(loaded_distances, original_distances, atol=1e-6)


def _index_file_metadata(path: Path) -> dict[str, int | str]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def run_indexing(config: ProjectConfig) -> dict:
    """Build, reload, verify, benchmark, and report both FAISS indexes."""
    indexing = config.indexing
    faiss.omp_set_num_threads(indexing.omp_threads)
    retrieval_run_id, retrieval_run_dir = _latest_retrieval_run(
        config.retrieval.model_dir
    )
    ids_path = retrieval_run_dir / "candidate_movie_ids.npy"
    vectors_path = retrieval_run_dir / "item_embeddings.npy"
    movie_ids, vectors = validate_index_inputs(
        np.load(ids_path, allow_pickle=False),
        np.load(vectors_path, allow_pickle=False),
    )
    if indexing.k > len(movie_ids):
        raise ValueError("configured k exceeds the number of candidate movies")
    query_user_ids, queries = _make_user_queries(
        retrieval_run_dir, indexing.query_count, indexing.random_seed
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{retrieval_run_id}-faiss-{timestamp}"
    run_dir = indexing.index_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    np.save(run_dir / "benchmark_user_ids.npy", query_user_ids)

    started = time.perf_counter()
    exact_index = build_exact_index(movie_ids, vectors)
    exact_build_seconds = time.perf_counter() - started
    exact_path = run_dir / "exact_ip.faiss"
    exact_reloaded = write_and_reload(exact_index, exact_path)
    _verify_reload(exact_index, exact_reloaded, queries, indexing.k)
    exact_metrics, exact_ids = benchmark_index(
        exact_reloaded,
        queries,
        indexing.k,
        indexing.warmup_queries,
        indexing.benchmark_repeats,
    )

    started = time.perf_counter()
    hnsw_index = build_hnsw_index(
        movie_ids,
        vectors,
        indexing.hnsw_m,
        indexing.hnsw_ef_construction,
    )
    hnsw_build_seconds = time.perf_counter() - started
    operating_points: list[dict] = []
    for ef_search in indexing.hnsw_ef_search_values:
        set_hnsw_ef_search(hnsw_index, ef_search)
        performance, approximate_ids = benchmark_index(
            hnsw_index,
            queries,
            indexing.k,
            indexing.warmup_queries,
            indexing.benchmark_repeats,
        )
        operating_points.append(
            {
                "ef_search": ef_search,
                "recall_at_k_vs_exact": approximate_recall_at_k(
                    exact_ids, approximate_ids
                ),
                **performance,
            }
        )

    eligible = [
        point for point in operating_points if point["recall_at_k_vs_exact"] >= 0.95
    ]
    selected = eligible[0] if eligible else operating_points[-1]
    set_hnsw_ef_search(hnsw_index, selected["ef_search"])
    hnsw_path = run_dir / "hnsw_ip.faiss"
    hnsw_reloaded = write_and_reload(hnsw_index, hnsw_path)
    set_hnsw_ef_search(hnsw_reloaded, selected["ef_search"])
    _verify_reload(hnsw_index, hnsw_reloaded, queries, indexing.k)

    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_retrieval_run_id": retrieval_run_id,
        "source": {
            "candidate_movie_ids": str(ids_path),
            "candidate_movie_ids_sha256": _file_sha256(ids_path),
            "item_embeddings": str(vectors_path),
            "item_embeddings_sha256": _file_sha256(vectors_path),
        },
        "library": {"faiss": faiss.__version__, "metric": "inner_product"},
        "vectors": {
            "count": len(movie_ids),
            "dimension": vectors.shape[1],
            "dtype": str(vectors.dtype),
            "l2_norm_min": float(np.linalg.norm(vectors, axis=1).min()),
            "l2_norm_max": float(np.linalg.norm(vectors, axis=1).max()),
        },
        "benchmark": {
            "query_source": "seeded known users passed through exported user tower",
            "query_count": len(queries),
            "k": indexing.k,
            "omp_threads": indexing.omp_threads,
            "warmup_queries": indexing.warmup_queries,
            "repeats": indexing.benchmark_repeats,
            "latency_scope": "in-process index.search call including Python binding",
        },
        "exact": {
            "type": "IndexIDMap2(IndexFlatIP)",
            "build_seconds": exact_build_seconds,
            "performance": exact_metrics,
            "artifact": _index_file_metadata(exact_path),
            "reload_verified": True,
        },
        "hnsw": {
            "type": "IndexIDMap2(IndexHNSWFlat)",
            "m": indexing.hnsw_m,
            "ef_construction": indexing.hnsw_ef_construction,
            "build_seconds": hnsw_build_seconds,
            "operating_points": operating_points,
            "selected_ef_search": selected["ef_search"],
            "selection_rule": "smallest measured efSearch with Recall@K >= 0.95",
            "artifact": _index_file_metadata(hnsw_path),
            "reload_verified": True,
        },
    }
    report_path = run_dir / "index_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    indexing.index_dir.mkdir(parents=True, exist_ok=True)
    (indexing.index_dir / "latest.json").write_text(
        json.dumps(
            {"run_id": run_id, "run_dir": str(run_dir), "report": str(report_path)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.data.reports_dir / "faiss_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        name: value
        for name, value in {"query_count": args.query_count, "k": args.k}.items()
        if value is not None
    }
    if overrides:
        indexing = replace(config.indexing, **overrides)
        indexing.validate()
        config = replace(config, indexing=indexing)
    print(json.dumps(run_indexing(config), indent=2))


if __name__ == "__main__":
    main()
