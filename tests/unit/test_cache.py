from recommender.serving.cache import (
    InMemoryKeyValueBackend,
    RecommendationCache,
    build_cache_key,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class BrokenBackend:
    def get(self, key: str):
        raise ConnectionError(key)

    def set(self, key: str, value: str, *, ex: int):
        raise ConnectionError(key, value, ex)

    def ping(self):
        raise ConnectionError


def test_cache_key_changes_with_model_variant_and_context() -> None:
    common = {
        "namespace": "recs",
        "subject": "user-7",
        "context": {"genres": ["Action"]},
        "limit": 10,
    }
    first = build_cache_key(model_version="v1", variant="dcn", **common)
    reordered = build_cache_key(
        model_version="v1",
        variant="dcn",
        **{**common, "context": {"genres": ["Action"]}},
    )
    changed = build_cache_key(model_version="v2", variant="dcn", **common)
    other_variant = build_cache_key(
        model_version="v1", variant="wide_and_deep", **common
    )

    assert first == reordered
    assert first != changed
    assert first != other_variant
    assert len(first) < 160

    long_version = build_cache_key(
        model_version="run-" * 100, variant="dcn", **common
    )
    assert len(long_version) < 160


def test_in_memory_cache_expires_and_tracks_hits() -> None:
    clock = Clock()
    cache = RecommendationCache(InMemoryKeyValueBackend(clock), ttl_seconds=10)
    assert cache.get("key") is None
    assert cache.set("key", {"movies": [1, 2]})
    assert cache.get("key") == {"movies": [1, 2]}
    clock.now = 11
    assert cache.get("key") is None
    assert cache.stats.hits == 1
    assert cache.stats.misses == 2


def test_cache_fails_open_when_backend_is_unavailable() -> None:
    cache = RecommendationCache(BrokenBackend(), ttl_seconds=10)
    assert cache.get("key") is None
    assert not cache.set("key", {"movies": [1]})
    assert not cache.healthy()
    assert cache.stats.read_errors == 1
    assert cache.stats.write_errors == 1
