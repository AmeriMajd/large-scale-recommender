"""Versioned recommendation caching with Redis and fail-open behavior."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol


class KeyValueBackend(Protocol):
    def get(self, key: str) -> str | bytes | None: ...

    def set(self, key: str, value: str, *, ex: int) -> Any: ...

    def ping(self) -> Any: ...


def build_cache_key(
    *,
    namespace: str,
    model_version: str,
    variant: str,
    subject: str,
    context: dict,
    limit: int,
) -> str:
    """Build a bounded deterministic key containing every output dependency."""
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
    context_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    version_hash = hashlib.sha256(model_version.encode("utf-8")).hexdigest()[:16]
    return ":".join(
        [
            namespace,
            "recommendations",
            version_hash,
            variant,
            subject,
            str(limit),
            context_hash,
        ]
    )


class InMemoryKeyValueBackend:
    """Redis-like TTL backend for tests and local demonstrations."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._values: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        value = self._values.get(key)
        if value is None:
            return None
        expires_at, payload = value
        if expires_at <= self._clock():
            self._values.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: str, *, ex: int) -> bool:
        self._values[key] = (self._clock() + ex, value)
        return True

    def ping(self) -> bool:
        return True


class RedisKeyValueBackend:
    """Small redis-py adapter with bounded connection and command timeouts."""

    def __init__(self, url: str, timeout_seconds: float = 0.2):
        from redis import Redis

        self.client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, *, ex: int) -> bool:
        return bool(self.client.set(key, value, ex=ex))

    def ping(self) -> bool:
        return bool(self.client.ping())


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    read_errors: int = 0
    write_errors: int = 0


class RecommendationCache:
    """JSON recommendation cache that degrades safely when Redis is unavailable."""

    def __init__(self, backend: KeyValueBackend, ttl_seconds: int):
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()

    def get(self, key: str) -> dict | None:
        try:
            encoded = self.backend.get(key)
            if encoded is None:
                self.stats.misses += 1
                return None
            if isinstance(encoded, bytes):
                encoded = encoded.decode("utf-8")
            value = json.loads(encoded)
            if not isinstance(value, dict):
                raise ValueError("cached recommendation payload is not an object")
            self.stats.hits += 1
            return value
        except Exception:
            self.stats.read_errors += 1
            return None

    def set(self, key: str, value: dict) -> bool:
        try:
            encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
            self.backend.set(key, encoded, ex=self.ttl_seconds)
            return True
        except Exception:
            self.stats.write_errors += 1
            return False

    def healthy(self) -> bool:
        try:
            return bool(self.backend.ping())
        except Exception:
            return False
