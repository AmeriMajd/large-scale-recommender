"""Create leakage-controlled training pairs and future validation candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from recommender.config import ProjectConfig, load_config  # noqa: E402
from recommender.retrieval.data import (  # noqa: E402
    build_evaluation_users,
    load_train_vocabularies,
)
from recommender.spark import create_spark_session  # noqa: E402


def build_training_pairs(
    train_sample: DataFrame,
    full_train: DataFrame,
    item_features: DataFrame,
    positive_rating: float,
    max_positives: int,
    negatives_per_positive: int,
    seed: int,
) -> DataFrame:
    """Pair deterministic positive examples with unobserved catalog negatives."""
    positive_pairs = (
        full_train.filter(F.col("rating") >= positive_rating)
        .select("userId", "movieId")
        .distinct()
    )
    positives = (
        train_sample.filter(F.col("rating") >= positive_rating)
        .select("userId", "movieId")
        .distinct()
        .withColumn(
            "_sample_order",
            F.pmod(
                F.xxhash64("userId", "movieId", F.lit(seed)), F.lit(2_147_483_647)
            ),
        )
        .orderBy("_sample_order", "userId", "movieId")
        .limit(max_positives)
        .drop("_sample_order")
        .withColumn("label", F.lit(1.0))
    )

    candidate_window = Window.orderBy("movieId")
    candidates = (
        item_features.filter(
            F.col("is_seen_in_train") & (F.col("positive_interaction_count") > 0)
        )
        .select("movieId")
        .distinct()
        .withColumn(
            "_candidate_index", F.row_number().over(candidate_window) - F.lit(1)
        )
    )
    candidate_count = candidates.count()
    if candidate_count == 0:
        raise ValueError("ranking candidate catalog is empty")

    negative_slots = full_train.sparkSession.range(negatives_per_positive).select(
        F.col("id").alias("_negative_slot")
    )
    negatives = (
        positives.select("userId", F.col("movieId").alias("_positive_movie_id"))
        .crossJoin(negative_slots)
        .withColumn(
            "_candidate_index",
            F.pmod(
                F.xxhash64(
                    "userId",
                    "_positive_movie_id",
                    "_negative_slot",
                    F.lit(seed),
                ),
                F.lit(candidate_count),
            ),
        )
        .join(F.broadcast(candidates), on="_candidate_index", how="inner")
        .select("userId", "movieId")
        .join(positive_pairs, on=["userId", "movieId"], how="left_anti")
        .dropDuplicates(["userId", "movieId"])
        .withColumn("label", F.lit(0.0))
    )
    return positives.select("userId", "movieId", "label").unionByName(negatives)


def _latest_run(pointer: Path) -> tuple[str, Path]:
    if not pointer.is_file():
        raise FileNotFoundError(f"artifact pointer does not exist: {pointer}")
    value = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = Path(value["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"artifact run directory does not exist: {run_dir}")
    return str(value["run_id"]), run_dir


def write_validation_candidate_pairs(config: ProjectConfig, output_path: Path) -> dict:
    """Retrieve future-validation candidates and label them without future features."""
    retrieval_run_id, retrieval_run_dir = _latest_run(
        config.retrieval.model_dir / "latest.json"
    )
    _, index_run_dir = _latest_run(config.indexing.index_dir / "latest.json")
    index_report = json.loads(
        (index_run_dir / "index_report.json").read_text(encoding="utf-8")
    )
    exact_index = faiss.read_index(index_report["exact"]["artifact"]["path"])

    vocabularies = load_train_vocabularies(
        config.data.processed_dir / "features"
    )
    users = build_evaluation_users(
        config.data.processed_dir / "features" / "validation_context",
        config.data.processed_dir / "ratings_train",
        vocabularies.user_ids,
        vocabularies.movie_ids,
        config.baseline.positive_rating,
        config.ranking.max_validation_users,
        config.ranking.random_seed,
        config.ranking.parquet_batch_size,
    )
    user_ids = np.asarray([user.user_id for user in users], dtype=np.int64)
    signature = tf.saved_model.load(
        str(retrieval_run_dir / "user_tower")
    ).signatures["serving_default"]
    queries = np.ascontiguousarray(
        signature(user_id=tf.constant(user_ids))["embedding"].numpy(),
        dtype=np.float32,
    )

    columns: dict[str, list] = {
        "userId": [],
        "movieId": [],
        "label": [],
        "retrieval_rank": [],
        "retrieval_score": [],
        "total_relevant_count": [],
    }
    for user, query in zip(users, queries, strict=True):
        fetch_k = min(
            exact_index.ntotal,
            config.ranking.candidate_k + len(user.seen_items),
        )
        distances, ids = exact_index.search(query.reshape(1, -1), fetch_k)
        retained = [
            (int(movie_id), float(score))
            for movie_id, score in zip(ids[0], distances[0], strict=True)
            if int(movie_id) not in user.seen_items
        ][: config.ranking.candidate_k]
        if len(retained) != config.ranking.candidate_k:
            raise ValueError(f"user {user.user_id} has too few eligible candidates")
        for rank, (movie_id, score) in enumerate(retained, start=1):
            columns["userId"].append(user.user_id)
            columns["movieId"].append(movie_id)
            columns["label"].append(float(movie_id in user.relevant_items))
            columns["retrieval_rank"].append(rank)
            columns["retrieval_score"].append(score)
            columns["total_relevant_count"].append(len(user.relevant_items))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), output_path)
    return {
        "retrieval_run_id": retrieval_run_id,
        "users": len(users),
        "rows": len(columns["userId"]),
        "positive_candidate_rows": int(sum(columns["label"])),
        "total_future_relevant_items": int(
            sum(len(user.relevant_items) for user in users)
        ),
    }


def add_snapshot_features(
    pairs: DataFrame, user_features: DataFrame, item_features: DataFrame
) -> DataFrame:
    """Join train-cutoff features and scale numeric columns into stable ranges."""
    user_columns = user_features.select(
        "userId",
        F.log1p("interaction_count").alias("_user_log_interactions"),
        F.coalesce("mean_rating", F.lit(0.0)).alias("_user_mean_rating"),
        F.coalesce("positive_rate", F.lit(0.0)).alias("user_positive_rate"),
        F.log1p(F.greatest("days_since_last_interaction", F.lit(0))).alias(
            "_user_log_recency"
        ),
    )
    item_columns = item_features.select(
        "movieId",
        F.log1p("interaction_count").alias("_item_log_interactions"),
        F.coalesce("mean_rating", F.lit(0.0)).alias("_item_mean_rating"),
        F.coalesce("positive_rate", F.lit(0.0)).alias("item_positive_rate"),
        (F.lit(1.0) - F.coalesce("popularity_percentile", F.lit(1.0))).alias(
            "item_popularity_score"
        ),
        F.log1p(F.greatest(F.coalesce("movie_age_years", F.lit(0)), F.lit(0))).alias(
            "_item_log_age"
        ),
    )
    return (
        pairs.join(user_columns, on="userId", how="inner")
        .join(item_columns, on="movieId", how="inner")
        .withColumn("user_log_interactions", F.col("_user_log_interactions") / 15.0)
        .withColumn("user_mean_rating", F.col("_user_mean_rating") / 5.0)
        .withColumn("user_log_recency", F.col("_user_log_recency") / 12.0)
        .withColumn("item_log_interactions", F.col("_item_log_interactions") / 15.0)
        .withColumn("item_mean_rating", F.col("_item_mean_rating") / 5.0)
        .withColumn("item_log_age", F.col("_item_log_age") / 8.0)
        .drop(
            "_user_log_interactions",
            "_user_mean_rating",
            "_user_log_recency",
            "_item_log_interactions",
            "_item_mean_rating",
            "_item_log_age",
        )
    )


def run_preparation(config: ProjectConfig, spark: SparkSession) -> dict:
    """Create both ranking datasets and a reproducibility report."""
    processed = config.data.processed_dir
    feature_dir = processed / "features"
    ranking_dir = processed / "ranking"
    train_sample = spark.read.parquet(str(processed / "retrieval" / "train_sample"))
    full_train = spark.read.parquet(str(processed / "ratings_train"))
    user_features = spark.read.parquet(str(feature_dir / "user_features"))
    item_features = spark.read.parquet(str(feature_dir / "item_features"))

    training_pairs = build_training_pairs(
        train_sample,
        full_train,
        item_features,
        config.baseline.positive_rating,
        config.ranking.max_training_positives,
        config.ranking.negatives_per_positive,
        config.ranking.random_seed,
    )
    training_examples = add_snapshot_features(
        training_pairs, user_features, item_features
    )
    training_path = ranking_dir / "training_examples"
    training_examples.write.mode("overwrite").parquet(str(training_path))

    raw_validation_path = ranking_dir / "validation_candidates_raw.parquet"
    validation_profile = write_validation_candidate_pairs(config, raw_validation_path)
    validation_pairs = spark.read.parquet(str(raw_validation_path))
    validation_examples = add_snapshot_features(
        validation_pairs, user_features, item_features
    )
    validation_path = ranking_dir / "validation_examples"
    validation_examples.write.mode("overwrite").parquet(str(validation_path))

    train_counts = {
        row["label"]: row["count"]
        for row in training_examples.groupBy("label").count().collect()
    }
    report = {
        "training": {
            "path": str(training_path),
            "rows": int(sum(train_counts.values())),
            "positive_rows": int(train_counts.get(1.0, 0)),
            "negative_rows": int(train_counts.get(0.0, 0)),
            "negative_sampling": "stable hash into train-positive candidate catalog",
            "known_positive_pairs_excluded_from_negatives": True,
        },
        "validation": {"path": str(validation_path), **validation_profile},
        "feature_snapshot": "Day 2 training-cutoff tables only",
        "random_seed": config.ranking.random_seed,
    }
    ranking_dir.mkdir(parents=True, exist_ok=True)
    (ranking_dir / "preparation_report.json").write_text(
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
        print(json.dumps(run_preparation(config, spark), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
