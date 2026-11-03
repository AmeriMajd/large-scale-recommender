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
        spark=SparkConfig(**raw["spark"]),
    )
