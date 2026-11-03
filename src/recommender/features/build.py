"""Build leakage-safe user, item, genre, and cold-start feature tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from recommender.config import ProjectConfig, load_config
from recommender.spark import create_spark_session

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class FeatureTables:
    user_features: DataFrame
    item_features: DataFrame
    user_genre_features: DataFrame
    validation_context: DataFrame
    test_context: DataFrame


def build_user_features(
    train: DataFrame,
    train_cutoff_epoch: int,
    positive_rating: float,
    recent_window_days: int,
) -> DataFrame:
    """Create one training-only aggregate row per observed user."""
    recent_start = train_cutoff_epoch - recent_window_days * SECONDS_PER_DAY
    return (
        train.groupBy("userId")
        .agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("movieId").alias("distinct_movie_count"),
            F.avg("rating").alias("mean_rating"),
            F.coalesce(F.stddev_pop("rating"), F.lit(0.0)).alias("rating_stddev"),
            F.min("rating").alias("min_rating"),
            F.max("rating").alias("max_rating"),
            F.sum(F.when(F.col("rating") >= positive_rating, 1).otherwise(0)).alias(
                "positive_interaction_count"
            ),
            F.min("timestamp").alias("first_interaction_epoch"),
            F.max("timestamp").alias("last_interaction_epoch"),
            F.sum(F.when(F.col("timestamp") >= recent_start, 1).otherwise(0)).alias(
                "recent_interaction_count"
            ),
        )
        .withColumn(
            "positive_rate",
            F.col("positive_interaction_count") / F.col("interaction_count"),
        )
        .withColumn(
            "activity_span_days",
            F.floor(
                (F.col("last_interaction_epoch") - F.col("first_interaction_epoch"))
                / SECONDS_PER_DAY
            ),
        )
        .withColumn(
            "days_since_last_interaction",
            F.floor(
                (F.lit(train_cutoff_epoch) - F.col("last_interaction_epoch"))
                / SECONDS_PER_DAY
            ),
        )
        .withColumn("feature_cutoff_epoch", F.lit(train_cutoff_epoch))
    )


def build_item_features(
    train: DataFrame,
    movies: DataFrame,
    train_cutoff_epoch: int,
    positive_rating: float,
    popularity_buckets: int,
) -> DataFrame:
    """Create catalog features while fitting all behavioral values on training."""
    popularity_window = Window.orderBy(
        F.desc("interaction_count"), F.desc("mean_rating"), F.asc("movieId")
    )
    item_stats = (
        train.groupBy("movieId")
        .agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("userId").alias("distinct_user_count"),
            F.avg("rating").alias("mean_rating"),
            F.coalesce(F.stddev_pop("rating"), F.lit(0.0)).alias("rating_stddev"),
            F.sum(F.when(F.col("rating") >= positive_rating, 1).otherwise(0)).alias(
                "positive_interaction_count"
            ),
            F.min("timestamp").alias("first_interaction_epoch"),
            F.max("timestamp").alias("last_interaction_epoch"),
        )
        .withColumn(
            "positive_rate",
            F.col("positive_interaction_count") / F.col("interaction_count"),
        )
        .withColumn("popularity_rank", F.row_number().over(popularity_window))
        .withColumn("popularity_percentile", F.percent_rank().over(popularity_window))
        .withColumn(
            "popularity_bucket",
            F.ntile(popularity_buckets).over(popularity_window),
        )
        .withColumn(
            "days_since_last_interaction",
            F.floor(
                (F.lit(train_cutoff_epoch) - F.col("last_interaction_epoch"))
                / SECONDS_PER_DAY
            ),
        )
    )

    cutoff_year = datetime.fromtimestamp(train_cutoff_epoch, tz=UTC).year
    return (
        movies.join(item_stats, on="movieId", how="left")
        .withColumn("is_seen_in_train", F.col("interaction_count").isNotNull())
        .fillna(
            0,
            subset=[
                "interaction_count",
                "distinct_user_count",
                "positive_interaction_count",
            ],
        )
        .withColumn("release_year_missing", F.col("release_year").isNull())
        .withColumn(
            "is_future_release_at_cutoff",
            F.when(
                F.col("release_year").isNull(), F.lit(None).cast("boolean")
            ).otherwise(F.col("release_year") > cutoff_year),
        )
        .withColumn(
            "movie_age_years",
            F.when(
                F.col("release_year").isNotNull()
                & (F.col("release_year") <= cutoff_year),
                F.lit(cutoff_year) - F.col("release_year"),
            ),
        )
        .withColumn("feature_cutoff_epoch", F.lit(train_cutoff_epoch))
    )


def build_user_genre_features(
    train: DataFrame,
    movies: DataFrame,
    train_cutoff_epoch: int,
    positive_rating: float,
    recent_window_days: int,
) -> DataFrame:
    """Create normalized historical and recent genre preferences per user."""
    recent_start = train_cutoff_epoch - recent_window_days * SECONDS_PER_DAY
    genre_events = train.join(
        movies.select("movieId", "genre_list"), on="movieId", how="inner"
    ).select("userId", "rating", "timestamp", F.explode("genre_list").alias("genre"))
    user_window = Window.partitionBy("userId")
    return (
        genre_events.groupBy("userId", "genre")
        .agg(
            F.count("*").alias("genre_interaction_count"),
            F.avg("rating").alias("genre_mean_rating"),
            F.sum(F.when(F.col("rating") >= positive_rating, 1).otherwise(0)).alias(
                "genre_positive_count"
            ),
            F.sum(F.when(F.col("timestamp") >= recent_start, 1).otherwise(0)).alias(
                "recent_genre_interaction_count"
            ),
        )
        .withColumn(
            "genre_positive_rate",
            F.col("genre_positive_count") / F.col("genre_interaction_count"),
        )
        .withColumn(
            "total_genre_interaction_count",
            F.sum("genre_interaction_count").over(user_window),
        )
        .withColumn(
            "total_recent_genre_interaction_count",
            F.sum("recent_genre_interaction_count").over(user_window),
        )
        .withColumn(
            "genre_interaction_share",
            F.col("genre_interaction_count") / F.col("total_genre_interaction_count"),
        )
        .withColumn(
            "recent_genre_share",
            F.when(
                F.col("total_recent_genre_interaction_count") > 0,
                F.col("recent_genre_interaction_count")
                / F.col("total_recent_genre_interaction_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("feature_cutoff_epoch", F.lit(train_cutoff_epoch))
    )


def mark_cold_start(interactions: DataFrame, train: DataFrame) -> DataFrame:
    """Annotate later interactions using only training entity membership."""
    known_users = (
        train.select("userId").distinct().withColumn("_known_user", F.lit(True))
    )
    known_items = (
        train.select("movieId").distinct().withColumn("_known_item", F.lit(True))
    )
    return (
        interactions.join(known_users, on="userId", how="left")
        .join(known_items, on="movieId", how="left")
        .withColumn("is_new_user", F.col("_known_user").isNull())
        .withColumn("is_new_item", F.col("_known_item").isNull())
        .withColumn(
            "cold_start_type",
            F.when(F.col("is_new_user") & F.col("is_new_item"), "new_user_and_item")
            .when(F.col("is_new_user"), "new_user")
            .when(F.col("is_new_item"), "new_item")
            .otherwise("known_user_and_item"),
        )
        .drop("_known_user", "_known_item")
    )


def build_feature_tables(
    train: DataFrame,
    validation: DataFrame,
    test: DataFrame,
    movies: DataFrame,
    config: ProjectConfig,
    train_cutoff_epoch: int,
) -> FeatureTables:
    """Build all Day 2 tables with training-only learned statistics."""
    return FeatureTables(
        user_features=build_user_features(
            train,
            train_cutoff_epoch,
            config.baseline.positive_rating,
            config.features.recent_window_days,
        ),
        item_features=build_item_features(
            train,
            movies,
            train_cutoff_epoch,
            config.baseline.positive_rating,
            config.features.popularity_buckets,
        ),
        user_genre_features=build_user_genre_features(
            train,
            movies,
            train_cutoff_epoch,
            config.baseline.positive_rating,
            config.features.recent_window_days,
        ),
        validation_context=mark_cold_start(validation, train),
        test_context=mark_cold_start(test, train),
    )


def load_processed_tables(
    spark: SparkSession, processed_dir: Path
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, int]:
    """Load Day 1 outputs and the training cutoff used for this snapshot."""
    table_names = ["ratings_train", "ratings_validation", "ratings_test", "movies"]
    tables = []
    for name in table_names:
        path = processed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Processed table does not exist: {path}")
        tables.append(spark.read.parquet(str(path)))

    summary_path = processed_dir / "split_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Split summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (*tables, int(summary["train_cutoff_epoch"]))


def write_feature_tables(tables: FeatureTables, output_dir: Path) -> None:
    """Write each feature table to a stable Parquet path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FeatureTables.__dataclass_fields__:
        getattr(tables, name).write.mode("overwrite").parquet(str(output_dir / name))


def profile_cold_start_context(context: DataFrame) -> dict:
    """Count distinct entities and interaction rows in each cold-start segment."""
    interactions_by_type = {
        row["cold_start_type"]: row["count"]
        for row in context.groupBy("cold_start_type").count().collect()
    }
    counts = context.agg(
        F.count("*").alias("interactions"),
        F.countDistinct("userId").alias("distinct_users"),
        F.countDistinct("movieId").alias("distinct_items"),
        F.countDistinct(F.when(F.col("is_new_user"), F.col("userId"))).alias(
            "new_users"
        ),
        F.countDistinct(F.when(F.col("is_new_item"), F.col("movieId"))).alias(
            "new_items"
        ),
    ).first()
    return {
        "interactions": counts["interactions"],
        "distinct_users": counts["distinct_users"],
        "distinct_items": counts["distinct_items"],
        "new_users": counts["new_users"],
        "new_items": counts["new_items"],
        "interactions_by_type": interactions_by_type,
    }


def profile_feature_tables(
    tables: FeatureTables,
    train_cutoff_epoch: int,
    config: ProjectConfig,
) -> dict:
    """Collect compact quality statistics for the generated snapshot."""
    item_counts = tables.item_features.agg(
        F.count("*").alias("catalog_items"),
        F.sum(F.col("is_seen_in_train").cast("long")).alias("items_seen_in_train"),
        F.sum(F.col("release_year_missing").cast("long")).alias(
            "items_missing_release_year"
        ),
        F.sum(
            F.coalesce(F.col("is_future_release_at_cutoff").cast("long"), F.lit(0))
        ).alias("future_release_items_at_cutoff"),
    ).first()
    top_items = [
        {
            "movie_id": row["movieId"],
            "title": row["title"],
            "interaction_count": row["interaction_count"],
            "mean_rating": row["mean_rating"],
        }
        for row in tables.item_features.filter("is_seen_in_train")
        .orderBy("popularity_rank")
        .select("movieId", "title", "interaction_count", "mean_rating")
        .limit(10)
        .collect()
    ]
    return {
        "feature_snapshot_cutoff_epoch": train_cutoff_epoch,
        "feature_snapshot_cutoff_utc": datetime.fromtimestamp(
            train_cutoff_epoch, tz=UTC
        ).isoformat(),
        "positive_rating_threshold": config.baseline.positive_rating,
        "recent_window_days": config.features.recent_window_days,
        "popularity_buckets": config.features.popularity_buckets,
        "training_users": tables.user_features.count(),
        "catalog_items": item_counts["catalog_items"],
        "items_seen_in_train": item_counts["items_seen_in_train"],
        "items_unseen_in_train": (
            item_counts["catalog_items"] - item_counts["items_seen_in_train"]
        ),
        "items_missing_release_year": item_counts["items_missing_release_year"],
        "future_release_items_at_cutoff": item_counts["future_release_items_at_cutoff"],
        "user_genre_rows": tables.user_genre_features.count(),
        "validation_context": profile_cold_start_context(tables.validation_context),
        "test_context": profile_cold_start_context(tables.test_context),
        "top_training_items": top_items,
    }


def run_feature_pipeline(spark: SparkSession, config: ProjectConfig) -> dict:
    train, validation, test, movies, cutoff = load_processed_tables(
        spark, config.data.processed_dir
    )
    feature_dir = config.data.processed_dir / "features"
    train = train.cache()
    movies = movies.cache()
    try:
        train.count()
        movies.count()
        tables = build_feature_tables(train, validation, test, movies, config, cutoff)
        write_feature_tables(tables, feature_dir)
    finally:
        train.unpersist()
        movies.unpersist()

    persisted_tables = FeatureTables(
        **{
            name: spark.read.parquet(str(feature_dir / name))
            for name in FeatureTables.__dataclass_fields__
        }
    )
    report = profile_feature_tables(persisted_tables, cutoff, config)
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.data.reports_dir / "feature_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spark = create_spark_session(config.spark)
    try:
        print(json.dumps(run_feature_pipeline(spark, config), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
