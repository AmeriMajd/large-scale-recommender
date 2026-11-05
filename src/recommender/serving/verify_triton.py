"""Compare live Triton outputs with the local SavedModel signatures."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from recommender.config import load_config
from recommender.serving.inference import (
    LocalTensorFlowInference,
    TritonHttpInference,
)


def _latest(pointer: Path) -> Path:
    value = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(value["run_dir"])


def _timed(callable_, repetitions: int) -> list[float]:
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        callable_()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return values


def verify(repetitions: int = 20) -> dict:
    config = load_config()
    retrieval_dir = _latest(config.retrieval.model_dir / "latest.json")
    ranking_dir = _latest(config.ranking.model_dir / "latest.json")
    local = LocalTensorFlowInference.load(retrieval_dir / "user_tower", ranking_dir)
    triton = TritonHttpInference(
        config.serving.triton_http_url, config.serving.triton_timeout_seconds
    )
    health = triton.health()
    if not health["ready"]:
        raise RuntimeError(f"Triton is not ready: {health}")

    user_id = 1
    local_embedding = local.user_embedding(user_id)
    remote_embedding = triton.user_embedding(user_id)
    embedding_error = float(np.max(np.abs(local_embedding - remote_embedding)))

    rows = 8
    features = {
        "user_id": np.full(rows, user_id, dtype=np.int64),
        "movie_id": np.arange(1, rows + 1, dtype=np.int64),
        "user_log_interactions": np.full(rows, 0.5, dtype=np.float32),
        "user_mean_rating": np.full(rows, 0.8, dtype=np.float32),
        "user_positive_rate": np.full(rows, 0.6, dtype=np.float32),
        "user_log_recency": np.full(rows, 0.2, dtype=np.float32),
        "item_log_interactions": np.linspace(0.1, 0.8, rows, dtype=np.float32),
        "item_mean_rating": np.full(rows, 0.7, dtype=np.float32),
        "item_positive_rate": np.full(rows, 0.5, dtype=np.float32),
        "item_popularity_score": np.linspace(0.9, 0.2, rows, dtype=np.float32),
        "item_log_age": np.full(rows, 0.4, dtype=np.float32),
        "retrieval_score": np.linspace(0.7, 0.1, rows, dtype=np.float32),
    }
    ranker_errors = {}
    for variant in sorted(local.ranker_names):
        expected = local.ranker_logits(variant, features)
        actual = triton.ranker_logits(variant, features)
        ranker_errors[variant] = float(np.max(np.abs(expected - actual)))

    latencies = _timed(lambda: triton.user_embedding(user_id), repetitions)
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "triton_health": health,
        "user_embedding_max_abs_error": embedding_error,
        "ranker_max_abs_error": ranker_errors,
        "user_tower_http_latency_ms": {
            "samples": repetitions,
            "mean": statistics.fmean(latencies),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }


def main() -> None:
    report = verify()
    output = Path("artifacts/reports/triton_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
