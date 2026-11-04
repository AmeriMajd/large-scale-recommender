"""Typed loading for the project's standard-library TOML configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    quantile_error: float

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0; received {total}")
        if (
            min(
                self.train_fraction,
                self.validation_fraction,
                self.test_fraction,
            )
            <= 0
        ):
            raise ValueError("Every temporal split must have a positive fraction")
        if not 0 <= self.quantile_error <= 1:
            raise ValueError("quantile_error must be between 0 and 1")


@dataclass(frozen=True)
class BaselineConfig:
    positive_rating: float
    recommendation_sizes: tuple[int, ...]


@dataclass(frozen=True)
class FeatureConfig:
    recent_window_days: int
    popularity_buckets: int

    def validate(self) -> None:
        if self.recent_window_days <= 0:
            raise ValueError("recent_window_days must be positive")
        if self.popularity_buckets <= 1:
            raise ValueError("popularity_buckets must be greater than one")


@dataclass(frozen=True)
class RetrievalConfig:
    model_dir: Path
    embedding_dimension: int
    batch_size: int
    parquet_batch_size: int
    shuffle_buffer: int
    epochs: int
    learning_rate: float
    max_train_interactions: int
    max_validation_users: int
    evaluation_ks: tuple[int, ...]
    random_seed: int
    sampling_modulus: int
    sampling_buckets: int

    def validate(self) -> None:
        positive_values = {
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.batch_size,
            "parquet_batch_size": self.parquet_batch_size,
            "shuffle_buffer": self.shuffle_buffer,
            "epochs": self.epochs,
            "max_train_interactions": self.max_train_interactions,
            "max_validation_users": self.max_validation_users,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.evaluation_ks or min(self.evaluation_ks) <= 0:
            raise ValueError("evaluation_ks must contain positive integers")
        if self.sampling_modulus <= 1:
            raise ValueError("sampling_modulus must be greater than one")
        if not 0 < self.sampling_buckets < self.sampling_modulus:
            raise ValueError("sampling_buckets must be between zero and the modulus")


@dataclass(frozen=True)
class IndexingConfig:
    index_dir: Path
    query_count: int
    k: int
    warmup_queries: int
    benchmark_repeats: int
    omp_threads: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search_values: tuple[int, ...]
    random_seed: int

    def validate(self) -> None:
        positive_values = {
            "query_count": self.query_count,
            "k": self.k,
            "benchmark_repeats": self.benchmark_repeats,
            "omp_threads": self.omp_threads,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries must not be negative")
        if not self.hnsw_ef_search_values or min(self.hnsw_ef_search_values) <= 0:
            raise ValueError("hnsw_ef_search_values must contain positive integers")
        if tuple(sorted(set(self.hnsw_ef_search_values))) != (
            self.hnsw_ef_search_values
        ):
            raise ValueError("hnsw_ef_search_values must be unique and increasing")


@dataclass(frozen=True)
class RankingConfig:
    model_dir: Path
    max_training_positives: int
    negatives_per_positive: int
    candidate_k: int
    max_validation_users: int
    parquet_batch_size: int
    batch_size: int
    epochs: int
    early_stopping_patience: int
    learning_rate: float
    id_embedding_dimension: int
    deep_layer_sizes: tuple[int, ...]
    cross_layers: int
    evaluation_ks: tuple[int, ...]
    random_seed: int

    def validate(self) -> None:
        positive_values = {
            "max_training_positives": self.max_training_positives,
            "negatives_per_positive": self.negatives_per_positive,
            "candidate_k": self.candidate_k,
            "max_validation_users": self.max_validation_users,
            "parquet_batch_size": self.parquet_batch_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "id_embedding_dimension": self.id_embedding_dimension,
            "cross_layers": self.cross_layers,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must not be negative")
        if not self.deep_layer_sizes or min(self.deep_layer_sizes) <= 0:
            raise ValueError("deep_layer_sizes must contain positive integers")
        if not self.evaluation_ks or min(self.evaluation_ks) <= 0:
            raise ValueError("evaluation_ks must contain positive integers")
        if max(self.evaluation_ks) > self.candidate_k:
            raise ValueError("ranking evaluation K cannot exceed candidate_k")


@dataclass(frozen=True)
class SparkConfig:
    app_name: str
    master: str
    shuffle_partitions: int
    driver_memory: str


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    split: SplitConfig
    baseline: BaselineConfig
    features: FeatureConfig
    retrieval: RetrievalConfig
    indexing: IndexingConfig
    ranking: RankingConfig
    spark: SparkConfig


def load_config(path: str | Path = "configs/base.toml") -> ProjectConfig:
    """Load and validate project configuration from a TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    split = SplitConfig(**raw["split"])
    split.validate()

    recommendation_sizes = tuple(raw["baseline"]["recommendation_sizes"])
    if not recommendation_sizes or min(recommendation_sizes) <= 0:
        raise ValueError("recommendation_sizes must contain positive integers")

    features = FeatureConfig(**raw["features"])
    features.validate()

    retrieval_raw = raw["retrieval"]
    retrieval = RetrievalConfig(
        model_dir=Path(retrieval_raw["model_dir"]),
        embedding_dimension=retrieval_raw["embedding_dimension"],
        batch_size=retrieval_raw["batch_size"],
        parquet_batch_size=retrieval_raw["parquet_batch_size"],
        shuffle_buffer=retrieval_raw["shuffle_buffer"],
        epochs=retrieval_raw["epochs"],
        learning_rate=retrieval_raw["learning_rate"],
        max_train_interactions=retrieval_raw["max_train_interactions"],
        max_validation_users=retrieval_raw["max_validation_users"],
        evaluation_ks=tuple(retrieval_raw["evaluation_ks"]),
        random_seed=retrieval_raw["random_seed"],
        sampling_modulus=retrieval_raw["sampling_modulus"],
        sampling_buckets=retrieval_raw["sampling_buckets"],
    )
    retrieval.validate()

    indexing_raw = raw["indexing"]
    indexing = IndexingConfig(
        index_dir=Path(indexing_raw["index_dir"]),
        query_count=indexing_raw["query_count"],
        k=indexing_raw["k"],
        warmup_queries=indexing_raw["warmup_queries"],
        benchmark_repeats=indexing_raw["benchmark_repeats"],
        omp_threads=indexing_raw["omp_threads"],
        hnsw_m=indexing_raw["hnsw_m"],
        hnsw_ef_construction=indexing_raw["hnsw_ef_construction"],
        hnsw_ef_search_values=tuple(indexing_raw["hnsw_ef_search_values"]),
        random_seed=indexing_raw["random_seed"],
    )
    indexing.validate()

    ranking_raw = raw["ranking"]
    ranking = RankingConfig(
        model_dir=Path(ranking_raw["model_dir"]),
        max_training_positives=ranking_raw["max_training_positives"],
        negatives_per_positive=ranking_raw["negatives_per_positive"],
        candidate_k=ranking_raw["candidate_k"],
        max_validation_users=ranking_raw["max_validation_users"],
        parquet_batch_size=ranking_raw["parquet_batch_size"],
        batch_size=ranking_raw["batch_size"],
        epochs=ranking_raw["epochs"],
        early_stopping_patience=ranking_raw["early_stopping_patience"],
        learning_rate=ranking_raw["learning_rate"],
        id_embedding_dimension=ranking_raw["id_embedding_dimension"],
        deep_layer_sizes=tuple(ranking_raw["deep_layer_sizes"]),
        cross_layers=ranking_raw["cross_layers"],
        evaluation_ks=tuple(ranking_raw["evaluation_ks"]),
        random_seed=ranking_raw["random_seed"],
    )
    ranking.validate()

    return ProjectConfig(
        data=DataConfig(
            raw_dir=Path(raw["data"]["raw_dir"]),
            processed_dir=Path(raw["data"]["processed_dir"]),
            reports_dir=Path(raw["data"]["reports_dir"]),
        ),
        split=split,
        baseline=BaselineConfig(
            positive_rating=raw["baseline"]["positive_rating"],
            recommendation_sizes=recommendation_sizes,
        ),
        features=features,
        retrieval=retrieval,
        indexing=indexing,
        ranking=ranking,
        spark=SparkConfig(**raw["spark"]),
    )
