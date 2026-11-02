"""Clean MovieLens data and create global chronological splits with Spark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from recommender.config import ProjectConfig, SplitConfig, load_config
from recommender.data.schemas import MOVIES_SCHEMA, RATINGS_SCHEMA
from recommender.spark import create_spark_session


@dataclass(frozen=True)
class SplitSummary:
    train_cutoff_epoch: int
    validation_cutoff_epoch: int
    train_rows: int
    validation_rows: int
    test_rows: int
    total_rows: int


def read_raw_tables(spark: SparkSession, raw_dir: Path) -> tuple[DataFrame, DataFrame]:
    """Read ratings and movies using explicit schemas and fail-fast CSV parsing."""
    ratings_path = raw_dir / "ratings.csv"
    movies_path = raw_dir / "movies.csv"
    for path in (ratings_path, movies_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required MovieLens file does not exist: {path}")

    options = {"header": "true", "mode": "FAILFAST", "enforceSchema": "true"}
    ratings = (
        spark.read.options(**options).schema(RATINGS_SCHEMA).csv(str(ratings_path))
    )
    movies = spark.read.options(**options).schema(MOVIES_SCHEMA).csv(str(movies_path))
    return ratings, movies


def clean_ratings(ratings: DataFrame) -> DataFrame:
    """Apply the MovieLens rating contract and add a UTC event timestamp."""
    return (
        ratings.filter(
            (F.col("userId") > 0)
            & (F.col("movieId") > 0)
            & F.col("rating").between(0.5, 5.0)
            & (F.col("timestamp") > 0)
        )
        .dropDuplicates(["userId", "movieId", "rating", "timestamp"])
        .withColumn("event_time", F.timestamp_seconds("timestamp"))
    )


def clean_movies(movies: DataFrame) -> DataFrame:
    """Parse genres and the optional release year while retaining source fields."""
    empty_genres = F.array().cast("array<string>")
    release_year_text = F.regexp_extract("title", r"\((\d{4})\)\s*$", 1)
    return (
        movies.filter(
            (F.col("movieId") > 0)
            & F.col("title").isNotNull()
            & F.col("genres").isNotNull()
        )
        .dropDuplicates(["movieId"])
        .withColumn(
            "genre_list",
            F.when(F.col("genres") == "(no genres listed)", empty_genres).otherwise(
                F.split("genres", r"\|")
            ),
        )
        .withColumn(
            "release_year",
            F.when(F.length(release_year_text) > 0, release_year_text.cast("integer")),
        )
    )


def temporal_split(
    ratings: DataFrame, config: SplitConfig
) -> tuple[DataFrame, DataFrame, DataFrame, tuple[int, int]]:
    """Split all interactions using global timestamp quantiles.

    Global cutoffs simulate training at one point in time and evaluating only on
    later events. Equal timestamps remain together to avoid one moment leaking
    across adjacent splits.
    """
    train_quantile = config.train_fraction
    validation_quantile = config.train_fraction + config.validation_fraction
    cutoffs = ratings.approxQuantile(
        "timestamp",
        [train_quantile, validation_quantile],
        config.quantile_error,
    )
    if len(cutoffs) != 2:
        raise ValueError("Could not calculate temporal cutoffs from an empty dataset")

    train_cutoff, validation_cutoff = (int(cutoffs[0]), int(cutoffs[1]))
    if train_cutoff >= validation_cutoff:
        raise ValueError(
            "Temporal cutoffs are not distinct; the dataset has too few timestamps"
        )

    train = ratings.filter(F.col("timestamp") <= train_cutoff)
    validation = ratings.filter(
        (F.col("timestamp") > train_cutoff) & (F.col("timestamp") <= validation_cutoff)
    )
    test = ratings.filter(F.col("timestamp") > validation_cutoff)
    return train, validation, test, (train_cutoff, validation_cutoff)


def assert_temporal_order(
    train: DataFrame, validation: DataFrame, test: DataFrame
) -> None:
    """Fail if any later split overlaps the time range of an earlier split."""
    boundaries = []
    for split in (train, validation, test):
        row = split.agg(
            F.min("timestamp").alias("min"), F.max("timestamp").alias("max")
        ).first()
        if row is None or row["min"] is None:
            raise ValueError("Every temporal split must contain at least one row")
        boundaries.append((row["min"], row["max"]))

    if not boundaries[0][1] < boundaries[1][0] <= boundaries[1][1] < boundaries[2][0]:
        raise AssertionError(f"Temporal split overlap detected: {boundaries}")


def write_processed_data(
    movies: DataFrame,
    train: DataFrame,
    validation: DataFrame,
    test: DataFrame,
    output_dir: Path,
) -> None:
    """Write canonical tables as compressed Parquet datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "movies": movies,
        "ratings_train": train,
        "ratings_validation": validation,
        "ratings_test": test,
    }
    for name, table in tables.items():
        table.write.mode("overwrite").parquet(str(output_dir / name))


def prepare_dataset(spark: SparkSession, config: ProjectConfig) -> SplitSummary:
    """Execute the complete Day 1 cleaning and temporal-split pipeline."""
    raw_ratings, raw_movies = read_raw_tables(spark, config.data.raw_dir)
    ratings = clean_ratings(raw_ratings).cache()
    movies = clean_movies(raw_movies)
    train, validation, test, cutoffs = temporal_split(ratings, config.split)
    assert_temporal_order(train, validation, test)

    counts = [split.count() for split in (train, validation, test)]
    summary = SplitSummary(
        train_cutoff_epoch=cutoffs[0],
        validation_cutoff_epoch=cutoffs[1],
        train_rows=counts[0],
        validation_rows=counts[1],
        test_rows=counts[2],
        total_rows=sum(counts),
    )
    write_processed_data(movies, train, validation, test, config.data.processed_dir)
    (config.data.processed_dir / "split_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    ratings.unpersist()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spark = create_spark_session(config.spark)
    try:
        summary = prepare_dataset(spark, config)
        print(json.dumps(asdict(summary), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
