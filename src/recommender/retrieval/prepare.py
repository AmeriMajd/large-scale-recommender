"""Create a deterministic, coverage-preserving retrieval training sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from recommender.config import ProjectConfig, load_config
from recommender.spark import create_spark_session


def build_retrieval_sample(
    train: DataFrame,
    positive_rating: float,
    sampling_modulus: int,
    sampling_buckets: int,
) -> DataFrame:
    """Keep user/item anchors plus a stable hash sample of positive events."""
    positive = train.filter(F.col("rating") >= positive_rating)
    user_window = Window.partitionBy("userId").orderBy(
        F.desc("timestamp"), F.asc("movieId")
    )
    item_window = Window.partitionBy("movieId").orderBy(
        F.desc("timestamp"), F.asc("userId")
    )
    return (
        positive.withColumn("_user_anchor_rank", F.row_number().over(user_window))
        .withColumn("_item_anchor_rank", F.row_number().over(item_window))
        .withColumn(
            "_hash_bucket",
            F.pmod(
                F.xxhash64("userId", "movieId", "timestamp"),
                F.lit(sampling_modulus),
            ),
        )
        .filter(
            (F.col("_user_anchor_rank") == 1)
            | (F.col("_item_anchor_rank") == 1)
            | (F.col("_hash_bucket") < sampling_buckets)
        )
        .drop("_user_anchor_rank", "_item_anchor_rank", "_hash_bucket")
    )


def _load_train(spark: SparkSession, processed_dir: Path) -> DataFrame:
    path = processed_dir / "ratings_train"
    if not path.exists():
        raise FileNotFoundError(f"Training table does not exist: {path}")
    return spark.read.parquet(str(path))


def run_prepare_retrieval(spark: SparkSession, config: ProjectConfig) -> dict:
    train = _load_train(spark, config.data.processed_dir)
    sample = build_retrieval_sample(
        train,
        config.baseline.positive_rating,
        config.retrieval.sampling_modulus,
        config.retrieval.sampling_buckets,
    )
    output_path = config.data.processed_dir / "retrieval" / "train_sample"
    sample.write.mode("overwrite").parquet(str(output_path))

    persisted = spark.read.parquet(str(output_path))
    counts = persisted.agg(
        F.count("*").alias("interactions"),
        F.countDistinct("userId").alias("distinct_users"),
        F.countDistinct("movieId").alias("distinct_items"),
        F.min("rating").alias("minimum_rating"),
        F.max("timestamp").alias("maximum_timestamp"),
    ).first()
    report = {
        "output_path": str(output_path),
        "positive_rating_threshold": config.baseline.positive_rating,
        "sampling_modulus": config.retrieval.sampling_modulus,
        "sampling_buckets": config.retrieval.sampling_buckets,
        "interactions": counts["interactions"],
        "distinct_users": counts["distinct_users"],
        "distinct_items": counts["distinct_items"],
        "minimum_rating": counts["minimum_rating"],
        "maximum_timestamp": counts["maximum_timestamp"],
    }
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.data.reports_dir / "retrieval_sample_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
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
        print(json.dumps(run_prepare_retrieval(spark, config), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
