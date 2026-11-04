"""Streaming ranking examples and retrieval-score feature computation."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

NUMERIC_FEATURES = (
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
MODEL_FEATURES = ("user_id", "movie_id", *NUMERIC_FEATURES)


def _files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return files


def _column(batch, name: str) -> np.ndarray:
    position = batch.schema.get_field_index(name)
    return batch.column(position).to_numpy(zero_copy_only=False)


class RetrievalScoreLookup:
    """Compute tower dot products from bounded in-memory embedding tables."""

    def __init__(self, retrieval_run_dir: Path, batch_size: int = 4096):
        self.user_ids = np.load(
            retrieval_run_dir / "user_ids.npy", allow_pickle=False
        ).astype(np.int64, copy=False)
        self.movie_ids = np.load(
            retrieval_run_dir / "candidate_movie_ids.npy", allow_pickle=False
        ).astype(np.int64, copy=False)
        self.item_embeddings = np.load(
            retrieval_run_dir / "item_embeddings.npy", allow_pickle=False
        ).astype(np.float32, copy=False)
        signature = tf.saved_model.load(
            str(retrieval_run_dir / "user_tower")
        ).signatures["serving_default"]
        parts = []
        for start in range(0, len(self.user_ids), batch_size):
            ids = tf.constant(self.user_ids[start : start + batch_size])
            parts.append(signature(user_id=ids)["embedding"].numpy())
        self.user_embeddings = np.ascontiguousarray(
            np.concatenate(parts), dtype=np.float32
        )

    def score(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        user_positions = np.searchsorted(self.user_ids, user_ids)
        movie_positions = np.searchsorted(self.movie_ids, movie_ids)
        if np.any(user_positions >= len(self.user_ids)) or np.any(
            movie_positions >= len(self.movie_ids)
        ):
            raise ValueError(
                "ranking examples contain IDs outside retrieval vocabulary"
            )
        users_match = np.array_equal(self.user_ids[user_positions], user_ids)
        movies_match = np.array_equal(self.movie_ids[movie_positions], movie_ids)
        if not users_match or not movies_match:
            raise ValueError(
                "ranking examples contain IDs outside retrieval vocabulary"
            )
        return np.einsum(
            "ij,ij->i",
            self.user_embeddings[user_positions],
            self.item_embeddings[movie_positions],
            optimize=True,
        ).astype(np.float32, copy=False)


def iter_example_batches(
    path: Path,
    scorer: RetrievalScoreLookup,
    parquet_batch_size: int,
    seed: int,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    """Yield model-ready batches without loading the example table at once."""
    files = _files(path)
    random.Random(seed).shuffle(files)
    base_columns = [
        "userId",
        "movieId",
        "label",
        *[name for name in NUMERIC_FEATURES if name != "retrieval_score"],
    ]
    dataset_schema = pads.dataset(str(files[0]), format="parquet").schema
    has_stored_score = "retrieval_score" in dataset_schema.names
    columns = [
        *base_columns,
        *(["retrieval_score"] if has_stored_score else []),
    ]
    for file in files:
        scanner = pads.dataset(str(file), format="parquet").scanner(
            columns=columns,
            batch_size=parquet_batch_size,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            user_ids = _column(batch, "userId").astype(np.int64, copy=False)
            movie_ids = _column(batch, "movieId").astype(np.int64, copy=False)
            features: dict[str, np.ndarray] = {
                "user_id": user_ids,
                "movie_id": movie_ids,
            }
            for name in NUMERIC_FEATURES:
                if name == "retrieval_score":
                    values = (
                        _column(batch, name).astype(np.float32, copy=False)
                        if has_stored_score
                        else scorer.score(user_ids, movie_ids)
                    )
                else:
                    values = _column(batch, name).astype(np.float32, copy=False)
                features[name] = values
            labels = _column(batch, "label").astype(np.float32, copy=False)
            yield features, labels


def make_ranking_dataset(
    path: Path,
    scorer: RetrievalScoreLookup,
    parquet_batch_size: int,
    batch_size: int,
    seed: int,
    shuffle_buffer: int = 100_000,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Create a bounded generator-backed TensorFlow ranking dataset."""
    feature_signature = {
        name: tf.TensorSpec(
            shape=(None,),
            dtype=tf.int64 if name in {"user_id", "movie_id"} else tf.float32,
        )
        for name in MODEL_FEATURES
    }

    def generator():
        return iter_example_batches(path, scorer, parquet_batch_size, seed)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(feature_signature, tf.TensorSpec((None,), tf.float32)),
    ).unbatch()
    if repeat:
        dataset = dataset.shuffle(
            shuffle_buffer, seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)
