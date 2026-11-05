"""End-to-end local recommendation orchestration for the demo API."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pyarrow.dataset as pads

from recommender.config import ProjectConfig, load_config
from recommender.serving.cache import (
    RecommendationCache,
    RedisKeyValueBackend,
    build_cache_key,
)
from recommender.serving.cold_start import (
    ColdStartRecommender,
    load_catalog,
)
from recommender.serving.experiment import (
    ExperimentStore,
    InMemoryExperimentStore,
    RedisExperimentStore,
    assign_variant,
)
from recommender.serving.inference import (
    InferenceBackend,
    LocalTensorFlowInference,
    TritonHttpInference,
)


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    score: float
    reason: str
    retrieval_score: float | None = None


def _latest_run(pointer: Path) -> tuple[str, Path, dict]:
    if not pointer.is_file():
        raise FileNotFoundError(f"artifact pointer does not exist: {pointer}")
    value = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = Path(value["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {run_dir}")
    return str(value["run_id"]), run_dir, value


class FeatureStore:
    """Bounded in-memory snapshot used to assemble ranker inputs."""

    def __init__(self, feature_dir: Path):
        user_table = pads.dataset(
            str(feature_dir / "user_features"), format="parquet"
        ).to_table(
            columns=[
                "userId",
                "interaction_count",
                "mean_rating",
                "positive_rate",
                "days_since_last_interaction",
            ]
        )
        item_table = pads.dataset(
            str(feature_dir / "item_features"), format="parquet"
        ).to_table(
            columns=[
                "movieId",
                "title",
                "interaction_count",
                "mean_rating",
                "positive_rate",
                "popularity_percentile",
                "movie_age_years",
            ]
        )
        self.users = {int(row["userId"]): row for row in user_table.to_pylist()}
        self.items = {int(row["movieId"]): row for row in item_table.to_pylist()}

    def title(self, movie_id: int) -> str:
        row = self.items.get(movie_id)
        return str(row["title"]) if row else f"Movie {movie_id}"

    def ranker_inputs(
        self,
        user_id: int,
        movie_ids: np.ndarray,
        retrieval_scores: np.ndarray,
    ) -> dict[str, np.ndarray]:
        user = self.users[user_id]
        item_rows = [self.items[int(movie_id)] for movie_id in movie_ids]
        count = len(movie_ids)

        def repeated(value: float) -> np.ndarray:
            return np.full(count, value, dtype=np.float32)

        user_recency = max(int(user["days_since_last_interaction"] or 0), 0)
        return {
            "user_id": np.full(count, user_id, dtype=np.int64),
            "movie_id": movie_ids.astype(np.int64, copy=False),
            "user_log_interactions": repeated(
                math.log1p(int(user["interaction_count"])) / 15.0
            ),
            "user_mean_rating": repeated(float(user["mean_rating"] or 0.0) / 5.0),
            "user_positive_rate": repeated(float(user["positive_rate"] or 0.0)),
            "user_log_recency": repeated(math.log1p(user_recency) / 12.0),
            "item_log_interactions": np.asarray(
                [math.log1p(int(row["interaction_count"])) / 15.0 for row in item_rows],
                dtype=np.float32,
            ),
            "item_mean_rating": np.asarray(
                [float(row["mean_rating"] or 0.0) / 5.0 for row in item_rows],
                dtype=np.float32,
            ),
            "item_positive_rate": np.asarray(
                [float(row["positive_rate"] or 0.0) for row in item_rows],
                dtype=np.float32,
            ),
            "item_popularity_score": np.asarray(
                [
                    1.0 - float(row["popularity_percentile"] or 0.0)
                    for row in item_rows
                ],
                dtype=np.float32,
            ),
            "item_log_age": np.asarray(
                [
                    math.log1p(max(int(row["movie_age_years"] or 0), 0)) / 8.0
                    for row in item_rows
                ],
                dtype=np.float32,
            ),
            "retrieval_score": retrieval_scores.astype(np.float32, copy=False),
        }


class SeenItemsRepository:
    """Read and memoize one user's full training history from Parquet."""

    def __init__(self, ratings_path: Path):
        self.dataset = pads.dataset(str(ratings_path), format="parquet")
        self.cache: dict[int, frozenset[int]] = {}

    def get(self, user_id: int) -> frozenset[int]:
        if user_id not in self.cache:
            table = self.dataset.to_table(
                columns=["movieId"], filter=pads.field("userId") == user_id
            )
            self.cache[user_id] = frozenset(
                int(value) for value in table["movieId"].to_pylist()
            )
        return self.cache[user_id]


class RecommendationEngine:
    """Retrieve, rank, cache, and log recommendations for known/cold users."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        cache: RecommendationCache,
        experiments: ExperimentStore,
        model_version: str,
        user_ids: np.ndarray,
        index: faiss.Index,
        inference: InferenceBackend,
        feature_store: FeatureStore,
        seen_items: SeenItemsRepository,
        cold_start: ColdStartRecommender,
    ) -> None:
        self.config = config
        self.cache = cache
        self.experiments = experiments
        self.model_version = model_version
        self.known_user_ids = frozenset(int(value) for value in user_ids)
        self.index = index
        self.inference = inference
        self.features = feature_store
        self.seen_items = seen_items
        self.cold_start = cold_start

    def _cache_key(
        self, variant: str, subject: str, context: dict, limit: int
    ) -> str:
        return build_cache_key(
            namespace=self.config.serving.model_namespace,
            model_version=self.model_version,
            variant=variant,
            subject=subject,
            context=context,
            limit=limit,
        )

    def _record_impression(
        self, variant: str, subject: str, recommendations: list[dict]
    ) -> str | None:
        try:
            experiment_id = (
                self.config.serving.experiment_id
                if variant in self.inference.ranker_names
                else "cold-start-fallback"
            )
            impression = self.experiments.record_impression(
                experiment_id=experiment_id,
                variant=variant,
                subject_id=subject,
                movie_ids=tuple(item["movie_id"] for item in recommendations),
                model_version=self.model_version,
            )
            return impression.impression_id
        except Exception:
            return None

    def _requested_limit(self, limit: int | None) -> int:
        value = limit or self.config.serving.recommendation_limit
        if value <= 0:
            raise ValueError("recommendation limit must be positive")
        if value > self.config.serving.retrieval_candidate_k:
            raise ValueError(
                "recommendation limit cannot exceed retrieval candidate count"
            )
        return value

    def _response(
        self,
        *,
        subject: str,
        variant: str,
        strategy: str,
        recommendations: list[dict],
        cache_hit: bool,
    ) -> dict:
        return {
            "subject": subject,
            "variant": variant,
            "strategy": strategy,
            "model_version": self.model_version,
            "cache_hit": cache_hit,
            "impression_id": self._record_impression(
                variant, subject, recommendations
            ),
            "recommendations": recommendations,
        }

    def recommend(self, user_id: int, limit: int | None = None) -> dict:
        """Recommend to a known user, or route an unknown user to popularity."""
        requested_limit = self._requested_limit(limit)
        if user_id not in self.known_user_ids:
            return self.recommend_cold_start(
                subject=f"user-{user_id}", limit=requested_limit
            )
        subject = f"user-{user_id}"
        variant = assign_variant(self.config.serving.experiment_id, subject)
        key = self._cache_key(variant, subject, {}, requested_limit)
        cached = self.cache.get(key)
        if cached:
            return self._response(
                subject=subject,
                variant=variant,
                strategy="two_tower_faiss_ranker",
                recommendations=cached["recommendations"],
                cache_hit=True,
            )

        seen = self.seen_items.get(user_id)
        embedding = self.inference.user_embedding(user_id)
        candidate_k = self.config.serving.retrieval_candidate_k
        fetch_k = min(self.index.ntotal, candidate_k + len(seen))
        distances, movie_ids = self.index.search(
            np.ascontiguousarray(embedding, dtype=np.float32), fetch_k
        )
        retained = [
            (int(movie_id), float(score))
            for movie_id, score in zip(movie_ids[0], distances[0], strict=True)
            if int(movie_id) not in seen and int(movie_id) in self.features.items
        ][:candidate_k]
        if len(retained) < requested_limit:
            raise RuntimeError("retrieval returned too few eligible candidates")
        candidate_ids = np.asarray([value[0] for value in retained], dtype=np.int64)
        retrieval_scores = np.asarray(
            [value[1] for value in retained], dtype=np.float32
        )
        features = self.features.ranker_inputs(
            user_id, candidate_ids, retrieval_scores
        )
        logits = self.inference.ranker_logits(variant, features)
        order = np.lexsort((candidate_ids, -logits))[:requested_limit]
        recommendations = [
            asdict(
                Recommendation(
                    movie_id=int(candidate_ids[position]),
                    title=self.features.title(int(candidate_ids[position])),
                    score=float(logits[position]),
                    reason=variant,
                    retrieval_score=float(retrieval_scores[position]),
                )
            )
            for position in order
        ]
        self.cache.set(key, {"recommendations": recommendations})
        return self._response(
            subject=subject,
            variant=variant,
            strategy="two_tower_faiss_ranker",
            recommendations=recommendations,
            cache_hit=False,
        )

    def recommend_cold_start(
        self,
        *,
        subject: str,
        selected_genres: tuple[str, ...] = (),
        liked_movie_ids: tuple[int, ...] = (),
        limit: int | None = None,
    ) -> dict:
        """Recommend without passing an unknown user through learned ID models."""
        requested_limit = self._requested_limit(limit)
        context = {
            "genres": sorted(set(selected_genres)),
            "liked_movie_ids": sorted(set(liked_movie_ids)),
        }
        variant = "cold_start"
        key = self._cache_key(variant, subject, context, requested_limit)
        cached = self.cache.get(key)
        if cached:
            return self._response(
                subject=subject,
                variant=variant,
                strategy=cached["strategy"],
                recommendations=cached["recommendations"],
                cache_hit=True,
            )
        results = self.cold_start.recommend(
            selected_genres=selected_genres,
            liked_movie_ids=liked_movie_ids,
            limit=requested_limit,
        )
        recommendations = [
            asdict(
                Recommendation(
                    movie_id=item.movie_id,
                    title=item.title,
                    score=item.score,
                    reason=item.reason,
                )
            )
            for item in results
        ]
        strategy = results[0].reason if results else "no_results"
        self.cache.set(
            key,
            {"strategy": strategy, "recommendations": recommendations},
        )
        return self._response(
            subject=subject,
            variant=variant,
            strategy=strategy,
            recommendations=recommendations,
            cache_hit=False,
        )

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_version": self.model_version,
            "known_users": len(self.known_user_ids),
            "indexed_items": self.index.ntotal,
            "cache_healthy": self.cache.healthy(),
            "inference": self.inference.health(),
        }


def build_default_engine(
    config: ProjectConfig | None = None,
) -> RecommendationEngine:
    """Load the latest successful artifacts and construct the local demo engine."""
    project = config or load_config()
    retrieval_id, retrieval_dir, _ = _latest_run(
        project.retrieval.model_dir / "latest.json"
    )
    _, index_dir, _ = _latest_run(project.indexing.index_dir / "latest.json")
    ranking_id, ranking_dir, _ = _latest_run(
        project.ranking.model_dir / "latest.json"
    )
    index_report = json.loads(
        (index_dir / "index_report.json").read_text(encoding="utf-8")
    )
    index = faiss.read_index(index_report["exact"]["artifact"]["path"])
    backend_name = os.getenv(
        "RECSYS_INFERENCE_BACKEND", project.serving.inference_backend
    )
    if backend_name == "local":
        inference: InferenceBackend = LocalTensorFlowInference.load(
            retrieval_dir / "user_tower", ranking_dir
        )
    elif backend_name == "triton":
        inference = TritonHttpInference(
            os.getenv("RECSYS_TRITON_HTTP_URL", project.serving.triton_http_url),
            project.serving.triton_timeout_seconds,
        )
        health = inference.health()
        if not health["ready"]:
            raise RuntimeError(f"Triton is not ready: {health}")
    else:
        raise ValueError("RECSYS_INFERENCE_BACKEND must be local or triton")
    redis_backend = RedisKeyValueBackend(
        os.getenv("RECSYS_REDIS_URL", project.serving.redis_url)
    )
    cache = RecommendationCache(redis_backend, project.serving.cache_ttl_seconds)
    if cache.healthy():
        experiments: ExperimentStore = RedisExperimentStore(
            redis_backend.client,
            project.serving.model_namespace,
            project.serving.event_stream_maxlen,
        )
    else:
        experiments = InMemoryExperimentStore()
    feature_dir = project.data.processed_dir / "features"
    return RecommendationEngine(
        config=project,
        cache=cache,
        experiments=experiments,
        model_version=f"{retrieval_id}+{ranking_id}",
        user_ids=np.load(retrieval_dir / "user_ids.npy", allow_pickle=False),
        index=index,
        inference=inference,
        feature_store=FeatureStore(feature_dir),
        seen_items=SeenItemsRepository(project.data.processed_dir / "ratings_train"),
        cold_start=ColdStartRecommender(load_catalog(feature_dir / "item_features")),
    )
