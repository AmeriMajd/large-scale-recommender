"""Deterministic A/B assignment and impression/click accounting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4


def assign_variant(
    experiment_id: str,
    subject_id: str,
    variants: tuple[str, ...] = ("wide_and_deep", "dcn"),
) -> str:
    """Map one subject stably and evenly into an ordered variant list."""
    if not experiment_id or not subject_id:
        raise ValueError("experiment and subject IDs must not be empty")
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("variants must contain at least two unique values")
    digest = hashlib.sha256(f"{experiment_id}:{subject_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % len(variants)
    return variants[bucket]


@dataclass(frozen=True)
class Impression:
    impression_id: str
    experiment_id: str
    variant: str
    subject_id: str
    movie_ids: tuple[int, ...]
    model_version: str
    created_at_utc: str


@dataclass(frozen=True)
class Click:
    impression_id: str
    movie_id: int
    rank: int
    created_at_utc: str


class ExperimentStore(Protocol):
    def record_impression(
        self,
        *,
        experiment_id: str,
        variant: str,
        subject_id: str,
        movie_ids: tuple[int, ...],
        model_version: str,
    ) -> Impression: ...

    def record_click(self, impression_id: str, movie_id: int) -> Click: ...

    def metrics(self, experiment_id: str) -> dict: ...


def _new_impression(
    experiment_id: str,
    variant: str,
    subject_id: str,
    movie_ids: tuple[int, ...],
    model_version: str,
) -> Impression:
    if not movie_ids:
        raise ValueError("an impression must contain at least one movie")
    return Impression(
        impression_id=str(uuid4()),
        experiment_id=experiment_id,
        variant=variant,
        subject_id=subject_id,
        movie_ids=movie_ids,
        model_version=model_version,
        created_at_utc=datetime.now(tz=UTC).isoformat(),
    )


def _click_for(impression: Impression, movie_id: int) -> Click:
    if movie_id not in impression.movie_ids:
        raise ValueError("clicked movie was not present in the impression")
    return Click(
        impression_id=impression.impression_id,
        movie_id=movie_id,
        rank=impression.movie_ids.index(movie_id) + 1,
        created_at_utc=datetime.now(tz=UTC).isoformat(),
    )


def summarize_experiment(
    experiment_id: str,
    impressions: list[Impression],
    clicks: list[Click],
) -> dict:
    """Calculate exposure, CTR, and reciprocal-rank-weighted CTR per variant."""
    matching = [item for item in impressions if item.experiment_id == experiment_id]
    by_id = {item.impression_id: item for item in matching}
    variants = sorted({item.variant for item in matching})
    first_click_by_impression = {}
    for click in clicks:
        first_click_by_impression.setdefault(click.impression_id, click)
    unique_clicks = list(first_click_by_impression.values())
    output = {}
    for variant in variants:
        variant_impressions = [item for item in matching if item.variant == variant]
        variant_clicks = [
            click
            for click in unique_clicks
            if click.impression_id in by_id
            and by_id[click.impression_id].variant == variant
        ]
        impression_count = len(variant_impressions)
        click_count = len(variant_clicks)
        output[variant] = {
            "impressions": impression_count,
            "clicks": click_count,
            "ctr": click_count / impression_count if impression_count else 0.0,
            "discounted_clicks_per_impression": (
                sum(1.0 / math.log2(click.rank + 1) for click in variant_clicks)
                / impression_count
                if impression_count
                else 0.0
            ),
        }
    return {
        "experiment_id": experiment_id,
        "variants": output,
        "total_impressions": len(matching),
        "total_clicks": sum(
            click.impression_id in by_id for click in unique_clicks
        ),
        "traffic_source": "demo_or_real_api_events_not_pre-labeled",
    }


class InMemoryExperimentStore:
    """Deterministic test/demo event store with click validation."""

    def __init__(self) -> None:
        self.impressions: dict[str, Impression] = {}
        self.clicks: list[Click] = []

    def record_impression(
        self,
        *,
        experiment_id: str,
        variant: str,
        subject_id: str,
        movie_ids: tuple[int, ...],
        model_version: str,
    ) -> Impression:
        impression = _new_impression(
            experiment_id, variant, subject_id, movie_ids, model_version
        )
        self.impressions[impression.impression_id] = impression
        return impression

    def record_click(self, impression_id: str, movie_id: int) -> Click:
        impression = self.impressions.get(impression_id)
        if impression is None:
            raise KeyError("unknown impression ID")
        click = _click_for(impression, movie_id)
        self.clicks.append(click)
        return click

    def metrics(self, experiment_id: str) -> dict:
        return summarize_experiment(
            experiment_id, list(self.impressions.values()), self.clicks
        )


class RedisExperimentStore:
    """Redis Streams event log with bounded retention and impression lookup."""

    def __init__(
        self,
        client: Any,
        namespace: str,
        maxlen: int,
        impression_ttl_seconds: int = 604_800,
    ) -> None:
        self.client = client
        self.stream_key = f"{namespace}:events"
        self.impression_prefix = f"{namespace}:impression:"
        self.maxlen = maxlen
        self.impression_ttl_seconds = impression_ttl_seconds

    def record_impression(
        self,
        *,
        experiment_id: str,
        variant: str,
        subject_id: str,
        movie_ids: tuple[int, ...],
        model_version: str,
    ) -> Impression:
        impression = _new_impression(
            experiment_id, variant, subject_id, movie_ids, model_version
        )
        encoded = json.dumps(asdict(impression), separators=(",", ":"))
        self.client.set(
            f"{self.impression_prefix}{impression.impression_id}",
            encoded,
            ex=self.impression_ttl_seconds,
        )
        self.client.xadd(
            self.stream_key,
            {"type": "impression", "payload": encoded},
            maxlen=self.maxlen,
            approximate=True,
        )
        return impression

    def record_click(self, impression_id: str, movie_id: int) -> Click:
        encoded = self.client.get(f"{self.impression_prefix}{impression_id}")
        if encoded is None:
            raise KeyError("unknown or expired impression ID")
        if isinstance(encoded, bytes):
            encoded = encoded.decode()
        value = json.loads(encoded)
        impression = Impression(
            **{**value, "movie_ids": tuple(value["movie_ids"])}
        )
        click = _click_for(impression, movie_id)
        self.client.xadd(
            self.stream_key,
            {"type": "click", "payload": json.dumps(asdict(click))},
            maxlen=self.maxlen,
            approximate=True,
        )
        return click

    def metrics(self, experiment_id: str) -> dict:
        impressions = []
        clicks = []
        for _, fields in self.client.xrange(self.stream_key):
            event_type = fields.get("type") or fields.get(b"type", b"").decode()
            payload = fields.get("payload") or fields.get(b"payload", b"").decode()
            value = json.loads(payload)
            if event_type == "impression":
                impressions.append(
                    Impression(**{**value, "movie_ids": tuple(value["movie_ids"])})
                )
            elif event_type == "click":
                clicks.append(Click(**value))
        return summarize_experiment(experiment_id, impressions, clicks)
