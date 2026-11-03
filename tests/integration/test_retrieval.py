from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from recommender.retrieval.data import (  # noqa: E402
    EvaluationUser,
    build_evaluation_users,
    iter_positive_interaction_batches,
    load_train_vocabularies,
    make_training_dataset,
    profile_training_sample,
)
from recommender.retrieval.model import build_model, export_tower  # noqa: E402
from recommender.retrieval.prepare import build_retrieval_sample  # noqa: E402


def _write_parquet(path: Path, name: str, values: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(values), path / f"{name}.parquet")


def test_streaming_sample_filters_positives_and_is_bounded(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings"
    _write_parquet(
        ratings,
        "part-000",
        {
            "userId": [1, 2, 3],
            "movieId": [10, 20, 30],
            "rating": [5.0, 2.0, 4.0],
        },
    )
    _write_parquet(
        ratings,
        "part-001",
        {
            "userId": [4, 5, 6],
            "movieId": [40, 50, 60],
            "rating": [4.5, 1.0, 5.0],
        },
    )

    batches = list(iter_positive_interaction_batches(ratings, 4.0, 2, 3, 42))
    user_ids = np.concatenate([batch["user_id"] for batch in batches])
    assert len(user_ids) == 3
    assert set(user_ids).issubset({1, 3, 4, 6})

    profile = profile_training_sample(ratings, 4.0, 2, 3, 42)
    assert profile["interactions"] == 3
    assert profile["distinct_users"] == 3
    assert profile["distinct_items"] == 3

    dataset = make_training_dataset(ratings, 4.0, 2, 3, 2, 3, 42)
    repeated_batch_sizes = [len(batch["user_id"]) for batch in dataset.take(4)]
    assert repeated_batch_sizes == [2, 1, 2, 1]


def test_spark_sample_preserves_positive_user_and_item_coverage(
    spark: SparkSession,
) -> None:
    train = spark.createDataFrame(
        [
            (1, 10, 5.0, 100),
            (1, 20, 4.0, 200),
            (2, 10, 4.5, 300),
            (3, 30, 3.5, 400),
            (4, 40, 5.0, 500),
        ],
        "userId long, movieId long, rating double, timestamp long",
    )
    sample = build_retrieval_sample(train, 4.0, 100, 1)
    rows = sample.select("userId", "movieId", "rating").collect()

    assert {row["userId"] for row in rows} == {1, 2, 4}
    assert {row["movieId"] for row in rows} == {10, 20, 40}
    assert min(row["rating"] for row in rows) >= 4.0


def test_vocabularies_and_evaluation_are_training_only(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_parquet(
        feature_dir / "user_features",
        "part-000",
        {"userId": [3, 1, 2], "positive_interaction_count": [2, 1, 0]},
    )
    _write_parquet(
        feature_dir / "item_features",
        "part-000",
        {
            "movieId": [10, 11, 12, 13],
            "positive_interaction_count": [1, 0, 2, 1],
            "is_seen_in_train": [True, False, True, True],
            "is_future_release_at_cutoff": [False, False, True, None],
        },
    )
    vocabularies = load_train_vocabularies(feature_dir)
    assert vocabularies.user_ids.tolist() == [1, 3]
    assert vocabularies.movie_ids.tolist() == [10, 12, 13]

    train = tmp_path / "train"
    _write_parquet(
        train,
        "part-000",
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 20],
            "rating": [5.0, 3.0, 4.0],
        },
    )
    validation = tmp_path / "validation"
    _write_parquet(
        validation,
        "part-000",
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 12, 21, 22, 12],
            "rating": [5.0, 5.0, 3.0, 5.0, 5.0],
            "is_new_user": [False, False, False, False, True],
            "is_new_item": [False, False, False, True, False],
        },
    )
    users = build_evaluation_users(
        validation,
        train,
        np.asarray([1, 2]),
        np.asarray([10, 11, 12, 20, 21]),
        positive_rating=4.0,
        max_users=10,
        seed=42,
        parquet_batch_size=2,
    )
    assert users == [
        EvaluationUser(
            user_id=1,
            relevant_items=frozenset({12}),
            seen_items=frozenset({10, 11}),
        )
    ]


def test_tfrs_model_trains_and_exports_a_tower(tmp_path: Path) -> None:
    model = build_model(
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([10, 20, 30], dtype=np.int64),
        embedding_dimension=4,
        learning_rate=0.1,
        seed=42,
    )
    loss = model.train_on_batch(
        {
            "user_id": np.asarray([1, 2], dtype=np.int64),
            "movie_id": np.asarray([10, 20], dtype=np.int64),
        }
    )
    assert np.all(np.isfinite(np.asarray(loss)))
    embeddings = model.user_tower(tf.constant([1, 999], dtype=tf.int64)).numpy()
    assert embeddings.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    export_path = tmp_path / "user_tower"
    export_tower(model.user_tower, export_path, "user_id")
    serving = tf.saved_model.load(str(export_path)).signatures["serving_default"]
    exported = serving(user_id=tf.constant([1], dtype=tf.int64))["embedding"]
    assert exported.shape == (1, 4)
