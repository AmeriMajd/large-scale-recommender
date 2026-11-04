from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from recommender.ranking.model import build_ranker, export_ranker  # noqa: E402
from recommender.ranking.prepare import build_training_pairs  # noqa: E402


def test_negative_sampling_excludes_every_known_positive(spark: SparkSession) -> None:
    sample = spark.createDataFrame(
        [(1, 10, 5.0), (2, 20, 4.0)], "userId long, movieId long, rating double"
    )
    full_train = spark.createDataFrame(
        [(1, 10, 5.0), (1, 20, 4.0), (2, 20, 4.0)],
        "userId long, movieId long, rating double",
    )
    items = spark.createDataFrame(
        [(10, True, 1), (20, True, 1), (30, True, 1), (40, True, 1)],
        "movieId long, is_seen_in_train boolean, positive_interaction_count long",
    )
    rows = build_training_pairs(sample, full_train, items, 4.0, 10, 8, 42).collect()

    known_positives = {(1, 10), (1, 20), (2, 20)}
    negative_pairs = {
        (row["userId"], row["movieId"]) for row in rows if row["label"] == 0.0
    }
    assert negative_pairs
    assert negative_pairs.isdisjoint(known_positives)


def _feature_batch() -> dict[str, np.ndarray]:
    numeric_names = (
        "user_log_interactions",
        "user_mean_rating",
        "user_positive_rate",
        "user_log_recency",
        "item_log_interactions",
        "item_mean_rating",
        "item_positive_rate",
        "item_popularity_score",
        "item_log_age",
        "retrieval_score",
    )
    features = {
        "user_id": np.asarray([1, 2, 1, 2], dtype=np.int64),
        "movie_id": np.asarray([10, 20, 20, 10], dtype=np.int64),
    }
    for index, name in enumerate(numeric_names):
        features[name] = np.asarray(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        ) + index / 100
    return features


def test_both_rankers_train_and_dcn_exports(tmp_path: Path) -> None:
    features = _feature_batch()
    labels = np.asarray([1, 1, 0, 0], dtype=np.float32)
    for kind in ("wide_and_deep", "dcn"):
        model = build_ranker(
            kind,
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([10, 20], dtype=np.int64),
            embedding_dimension=4,
            deep_layer_sizes=(8, 4),
            cross_layers=1,
            learning_rate=0.01,
            seed=42,
        )
        loss = model.train_on_batch(features, labels)
        assert np.all(np.isfinite(np.asarray(loss)))
        assert model(features).shape == (4,)

        if kind == "dcn":
            export_path = tmp_path / "dcn"
            export_ranker(model, export_path)
            signature = tf.saved_model.load(str(export_path)).signatures[
                "serving_default"
            ]
            inputs = {
                name: tf.constant(value) for name, value in features.items()
            }
            exported = signature(**inputs)
            assert exported["logit"].shape == (4,)
