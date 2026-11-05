import os
from uuid import uuid4

import pytest

from recommender.serving.cache import RecommendationCache, RedisKeyValueBackend
from recommender.serving.experiment import RedisExperimentStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REDIS_INTEGRATION") != "1",
    reason="set RUN_REDIS_INTEGRATION=1 with the local Redis service running",
)


def test_real_redis_cache_and_event_stream() -> None:
    backend = RedisKeyValueBackend(
        os.getenv("RECSYS_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    namespace = f"integration-{uuid4().hex}"
    cache_key = f"{namespace}:cache"
    cache = RecommendationCache(backend, ttl_seconds=30)
    store = RedisExperimentStore(backend.client, namespace, maxlen=100)
    impression_keys = []
    try:
        assert cache.healthy()
        assert cache.set(cache_key, {"movie_ids": [1, 2, 3]})
        assert cache.get(cache_key) == {"movie_ids": [1, 2, 3]}
        assert 0 < backend.client.ttl(cache_key) <= 30

        first = store.record_impression(
            experiment_id="integration-experiment",
            variant="dcn",
            subject_id="user-1",
            movie_ids=(10, 20),
            model_version="test-v1",
        )
        second = store.record_impression(
            experiment_id="integration-experiment",
            variant="dcn",
            subject_id="user-2",
            movie_ids=(20, 30),
            model_version="test-v1",
        )
        impression_keys.extend(
            [
                f"{store.impression_prefix}{first.impression_id}",
                f"{store.impression_prefix}{second.impression_id}",
            ]
        )
        click = store.record_click(first.impression_id, 20)
        metrics = store.metrics("integration-experiment")

        assert click.rank == 2
        assert metrics["variants"]["dcn"]["impressions"] == 2
        assert metrics["variants"]["dcn"]["ctr"] == 0.5
        assert backend.client.xlen(store.stream_key) == 3
    finally:
        backend.client.delete(cache_key, store.stream_key, *impression_keys)
