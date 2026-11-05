"""Popularity and genre-content fallbacks for users without learned embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.dataset as pads


@dataclass(frozen=True)
class CatalogItem:
    movie_id: int
    title: str
    genres: tuple[str, ...]
    popularity_score: float
    mean_rating: float
    interaction_count: int


@dataclass(frozen=True)
class FallbackResult:
    movie_id: int
    title: str
    score: float
    reason: str


def load_catalog(item_features_path: Path) -> list[CatalogItem]:
    """Load recommendable training-era catalog metadata from Day 2 Parquet."""
    table = pads.dataset(str(item_features_path), format="parquet").to_table(
        columns=[
            "movieId",
            "title",
            "genre_list",
            "interaction_count",
            "mean_rating",
            "popularity_percentile",
            "is_seen_in_train",
            "positive_interaction_count",
        ]
    )
    items = []
    for row in table.to_pylist():
        if not row["is_seen_in_train"] or row["positive_interaction_count"] <= 0:
            continue
        percentile = row["popularity_percentile"]
        items.append(
            CatalogItem(
                movie_id=int(row["movieId"]),
                title=str(row["title"]),
                genres=tuple(row["genre_list"] or ()),
                popularity_score=1.0 - float(percentile or 0.0),
                mean_rating=float(row["mean_rating"] or 0.0),
                interaction_count=int(row["interaction_count"]),
            )
        )
    if not items:
        raise ValueError("cold-start catalog cannot be empty")
    return items


class ColdStartRecommender:
    """Recommend with popularity and a temporary genre preference profile."""

    def __init__(self, items: list[CatalogItem], max_per_genre: int = 3):
        if not items:
            raise ValueError("cold-start catalog cannot be empty")
        if max_per_genre <= 0:
            raise ValueError("max_per_genre must be positive")
        self.items = tuple(items)
        self.by_id = {item.movie_id: item for item in items}
        if len(self.by_id) != len(items):
            raise ValueError("cold-start catalog movie IDs must be unique")
        self.max_per_genre = max_per_genre
        self.known_genres = {
            genre.casefold(): genre for item in items for genre in item.genres
        }

    @staticmethod
    def _diversity_bucket(item: CatalogItem) -> str:
        return min(item.genres, default="(no genres listed)")

    def _diversify(
        self, ranked: list[tuple[CatalogItem, float, str]], limit: int
    ) -> list[FallbackResult]:
        selected: list[tuple[CatalogItem, float, str]] = []
        deferred: list[tuple[CatalogItem, float, str]] = []
        counts: Counter[str] = Counter()
        for candidate in ranked:
            bucket = self._diversity_bucket(candidate[0])
            if counts[bucket] >= self.max_per_genre:
                deferred.append(candidate)
                continue
            selected.append(candidate)
            counts[bucket] += 1
            if len(selected) == limit:
                break
        if len(selected) < limit:
            chosen = {item.movie_id for item, _, _ in selected}
            selected.extend(
                candidate
                for candidate in deferred
                if candidate[0].movie_id not in chosen
            )
        return [
            FallbackResult(item.movie_id, item.title, float(score), reason)
            for item, score, reason in selected[:limit]
        ]

    def recommend(
        self,
        *,
        selected_genres: tuple[str, ...] = (),
        liked_movie_ids: tuple[int, ...] = (),
        exclude_movie_ids: frozenset[int] = frozenset(),
        limit: int = 10,
    ) -> list[FallbackResult]:
        """Return a deterministic diversified list for a cold-start request."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        weights: Counter[str] = Counter()
        for requested in selected_genres:
            canonical = self.known_genres.get(requested.casefold())
            if canonical:
                weights[canonical] += 2.0
        for movie_id in liked_movie_ids:
            item = self.by_id.get(movie_id)
            if not item or not item.genres:
                continue
            contribution = 1.0 / len(item.genres)
            for genre in item.genres:
                weights[genre] += contribution

        excluded = set(exclude_movie_ids).union(liked_movie_ids)
        maximum_weight = max(weights.values(), default=1.0)
        ranked: list[tuple[CatalogItem, float, str]] = []
        for item in self.items:
            if item.movie_id in excluded:
                continue
            quality = min(max(item.mean_rating / 5.0, 0.0), 1.0)
            if weights:
                affinity = sum(weights[genre] for genre in item.genres) / (
                    maximum_weight * max(len(item.genres), 1)
                )
                score = 0.65 * affinity + 0.25 * item.popularity_score + 0.10 * quality
                reason = "genre_content_popularity"
            else:
                score = 0.85 * item.popularity_score + 0.15 * quality
                reason = "diversified_popularity"
            ranked.append((item, score, reason))
        ranked.sort(key=lambda value: (-value[1], value[0].movie_id))
        return self._diversify(ranked, min(limit, len(ranked)))
