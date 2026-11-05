"""Exercise the real API and record local serving latency evidence."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from recommender.config import load_config
from recommender.serving.api import create_app
from recommender.serving.cache import InMemoryKeyValueBackend, RecommendationCache
from recommender.serving.engine import build_default_engine
from recommender.serving.experiment import InMemoryExperimentStore


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _timed_request(client: TestClient, method: str, path: str, **kwargs):
    started = time.perf_counter_ns()
    response = client.request(method, path, **kwargs)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    response.raise_for_status()
    return elapsed_ms, response.json()


def run_benchmark(repetitions: int = 30) -> dict:
    """Benchmark one cache miss and repeated cache hits through FastAPI."""
    if repetitions < 2:
        raise ValueError("benchmark repetitions must be at least two")
    config = load_config()
    engine = build_default_engine(config)
    redis_healthy = engine.cache.healthy()
    engine.cache = RecommendationCache(
        InMemoryKeyValueBackend(), config.serving.cache_ttl_seconds
    )
    engine.experiments = InMemoryExperimentStore()
    client = TestClient(create_app(engine))
    known_user = min(engine.known_user_ids)

    known_first_ms, known_first = _timed_request(
        client, "GET", f"/recommendations/{known_user}?limit=10"
    )
    known_hits = [
        _timed_request(
            client, "GET", f"/recommendations/{known_user}?limit=10"
        )[0]
        for _ in range(repetitions)
    ]
    cold_payload = {
        "session_id": "benchmark",
        "selected_genres": ["Action", "Comedy"],
        "liked_movie_ids": [],
        "limit": 10,
    }
    cold_first_ms, cold_first = _timed_request(
        client, "POST", "/recommendations/cold-start", json=cold_payload
    )
    cold_hits = [
        _timed_request(
            client, "POST", "/recommendations/cold-start", json=cold_payload
        )[0]
        for _ in range(repetitions)
    ]

    def summary(values: list[float]) -> dict:
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "samples": len(values),
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "scope": "local FastAPI TestClient; warm-cache latency is not production load",
        "model_version": engine.model_version,
        "redis_healthy_at_start": redis_healthy,
        "known_user": known_user,
        "known_user_variant": known_first["variant"],
        "known_user_cache_miss_ms": known_first_ms,
        "known_user_cache_hits": summary(known_hits),
        "cold_start_strategy": cold_first["strategy"],
        "cold_start_cache_miss_ms": cold_first_ms,
        "cold_start_cache_hits": summary(cold_hits),
        "recommendations_per_response": len(known_first["recommendations"]),
        "cache_stats": vars(engine.cache.stats),
    }


def main() -> None:
    report = run_benchmark()
    output = Path("artifacts/reports/serving_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
