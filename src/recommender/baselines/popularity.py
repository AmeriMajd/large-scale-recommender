"""Evaluate a train-only movie popularity baseline on future positive events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F

from recommender.config import ProjectConfig, load_config
from recommender.evaluation.metrics import ndcg_at_k, recall_at_k
from recommender.spark import create_spark_session


def popularity_ranking(train: DataFrame) -> list[int]:
    """Rank all training items by count, mean rating, then stable movie ID."""
    rows = (
        train.groupBy("movieId")
        .agg(
            F.count("*").alias("interaction_count"),
            F.avg("rating").alias("mean_rating"),
        )
        .orderBy(
            F.desc("interaction_count"),
            F.desc("mean_rating"),
            F.asc("movieId"),
        )
        .select("movieId")
        .collect()
    )
    return [int(row["movieId"]) for row in rows]


def recommend_popular(
    ranked_items: list[int], seen_items: set[int], k: int
) -> list[int]:
    """Return the first ``k`` globally popular movies the user has not seen."""
    recommendations = []
    for item_id in ranked_items:
        if item_id not in seen_items:
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
    return recommendations


def _evaluate_partition(rows, ranked_items: list[int], ks: tuple[int, ...]):
    max_k = max(ks)
    for row in rows:
        relevant = set(row["relevant_items"])
        seen = set(row["seen_items"] or [])
        recommendations = recommend_popular(ranked_items, seen, max_k)
        metrics = {
            "user_id": int(row["userId"]),
            "relevant_count": len(relevant),
            "recommendation_count": len(recommendations),
        }
        for k in ks:
            metrics[f"recall_at_{k}"] = recall_at_k(recommendations, relevant, k)
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(recommendations, relevant, k)
        yield Row(**metrics)


def evaluate_popularity(
    spark: SparkSession,
    train: DataFrame,
    test: DataFrame,
    positive_rating: float,
    ks: tuple[int, ...],
) -> dict[str, float | int]:
    """Evaluate popularity recommendations for users with future positives."""
    ranked_items = popularity_ranking(train)
    if not ranked_items:
        raise ValueError("Cannot evaluate popularity with an empty training split")

    truth = (
        test.filter(F.col("rating") >= positive_rating)
        .groupBy("userId")
        .agg(F.collect_set("movieId").alias("relevant_items"))
    )
    seen = train.groupBy("userId").agg(F.collect_set("movieId").alias("seen_items"))
    users = truth.join(seen, on="userId", how="left")
    ranked_broadcast = spark.sparkContext.broadcast(ranked_items)
    try:
        per_user = spark.createDataFrame(
            users.rdd.mapPartitions(
                lambda rows: _evaluate_partition(rows, ranked_broadcast.value, ks)
            )
        ).cache()
        metric_columns = [
            column for k in ks for column in (f"recall_at_{k}", f"ndcg_at_{k}")
        ]
        aggregate_expressions = [
            F.avg(column).alias(column) for column in metric_columns
        ]
        aggregate_expressions.extend(
            [
                F.count("*").alias("evaluated_users"),
                F.sum("relevant_count").alias("future_positive_events"),
                F.sum(
                    F.when(F.col("recommendation_count") < max(ks), 1).otherwise(0)
                ).alias("users_with_short_recommendation_lists"),
            ]
        )
        result = per_user.agg(*aggregate_expressions).first().asDict()
        per_user.unpersist()
    finally:
        ranked_broadcast.destroy()

    result["catalog_items_seen_in_train"] = len(ranked_items)
    result["positive_rating_threshold"] = positive_rating
    return result


def load_processed_tables(
    spark: SparkSession, processed_dir: Path
) -> tuple[DataFrame, DataFrame]:
    train_path = processed_dir / "ratings_train"
    test_path = processed_dir / "ratings_test"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Processed table does not exist: {path}")
    return spark.read.parquet(str(train_path)), spark.read.parquet(str(test_path))


def run_baseline(spark: SparkSession, config: ProjectConfig) -> dict[str, float | int]:
    train, test = load_processed_tables(spark, config.data.processed_dir)
    metrics = evaluate_popularity(
        spark,
        train,
        test,
        config.baseline.positive_rating,
        config.baseline.recommendation_sizes,
    )
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.data.reports_dir / "popularity_baseline.json"
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spark = create_spark_session(config.spark)
    try:
        print(json.dumps(run_baseline(spark, config), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
