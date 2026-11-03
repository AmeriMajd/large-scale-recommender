"""Streaming Parquet inputs and train-only retrieval vocabularies."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


@dataclass(frozen=True)
class RetrievalVocabularies:
    user_ids: np.ndarray
    movie_ids: np.ndarray


@dataclass(frozen=True)
class EvaluationUser:
    user_id: int
    relevant_items: frozenset[int]
    seen_items: frozenset[int]


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet part files found under {path}")
    return files


def _parquet_dataset(path: Path) -> pads.Dataset:
    return pads.dataset([str(file) for file in _parquet_files(path)], format="parquet")


def _column(batch, name: str) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    return batch.column(index).to_numpy(zero_copy_only=False)


def iter_positive_interaction_batches(
    ratings_path: Path,
    positive_rating: float,
    parquet_batch_size: int,
    max_interactions: int,
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield bounded positive interaction batches without collecting the table."""
    if parquet_batch_size <= 0 or max_interactions <= 0:
        raise ValueError("Batch size and maximum interactions must be positive")

    files = _parquet_files(ratings_path)
    random.Random(seed).shuffle(files)
    batch_iterators = [
        iter(
            pads.dataset(str(file), format="parquet")
            .scanner(
                columns=["userId", "movieId", "rating"],
                batch_size=parquet_batch_size,
                use_threads=False,
            )
            .to_batches()
        )
        for file in files
    ]
    emitted = 0
    while batch_iterators:
        remaining_iterators = []
        for batches in batch_iterators:
            try:
                batch = next(batches)
            except StopIteration:
                continue
            remaining_iterators.append(batches)
            ratings = _column(batch, "rating")
            positive = ratings >= positive_rating
            if not positive.any():
                continue

            remaining = max_interactions - emitted
            user_ids = _column(batch, "userId")[positive][:remaining]
            movie_ids = _column(batch, "movieId")[positive][:remaining]
            if user_ids.size:
                yield {
                    "user_id": user_ids.astype(np.int64, copy=False),
                    "movie_id": movie_ids.astype(np.int64, copy=False),
                }
                emitted += user_ids.size
            if emitted >= max_interactions:
                return
        batch_iterators = remaining_iterators


def make_training_dataset(
    ratings_path: Path,
    positive_rating: float,
    parquet_batch_size: int,
    max_interactions: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
):
    """Create a re-iterable ``tf.data`` pipeline over streamed Parquet batches."""
    import tensorflow as tf

    def generator() -> Iterator[dict[str, np.ndarray]]:
        return iter_positive_interaction_batches(
            ratings_path,
            positive_rating,
            parquet_batch_size,
            max_interactions,
            seed,
        )

    output_signature = {
        "user_id": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "movie_id": tf.TensorSpec(shape=(None,), dtype=tf.int64),
    }
    return (
        tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        .unbatch()
        .shuffle(
            min(shuffle_buffer, max_interactions),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )


def profile_training_sample(
    ratings_path: Path,
    positive_rating: float,
    parquet_batch_size: int,
    max_interactions: int,
    seed: int,
) -> dict[str, int]:
    """Count examples and vocabulary coverage in the deterministic sample."""
    users: set[int] = set()
    movies: set[int] = set()
    interactions = 0
    for batch in iter_positive_interaction_batches(
        ratings_path,
        positive_rating,
        parquet_batch_size,
        max_interactions,
        seed,
    ):
        interactions += len(batch["user_id"])
        users.update(int(value) for value in batch["user_id"])
        movies.update(int(value) for value in batch["movie_id"])
    return {
        "interactions": interactions,
        "distinct_users": len(users),
        "distinct_items": len(movies),
    }


def load_train_vocabularies(feature_dir: Path) -> RetrievalVocabularies:
    """Load deterministic user and candidate IDs fitted at the training cutoff."""
    user_batches = _parquet_dataset(feature_dir / "user_features").scanner(
        columns=["userId", "positive_interaction_count"], use_threads=False
    )
    user_parts = []
    for batch in user_batches.to_batches():
        positive = _column(batch, "positive_interaction_count") > 0
        user_parts.append(
            _column(batch, "userId")[positive].astype(np.int64, copy=False)
        )

    movie_parts: list[np.ndarray] = []
    item_batches = _parquet_dataset(feature_dir / "item_features").scanner(
        columns=["movieId", "positive_interaction_count", "is_seen_in_train"],
        use_threads=False,
    )
    for batch in item_batches.to_batches():
        movie_ids = _column(batch, "movieId")
        seen = _column(batch, "is_seen_in_train").astype(bool, copy=False)
        positive = _column(batch, "positive_interaction_count") > 0
        movie_parts.append(movie_ids[seen & positive].astype(np.int64, copy=False))

    if not user_parts or not movie_parts:
        raise ValueError("Training vocabularies cannot be empty")
    user_ids = np.unique(np.concatenate(user_parts))
    movie_ids = np.unique(np.concatenate(movie_parts))
    if not user_ids.size or not movie_ids.size:
        raise ValueError("Training vocabularies cannot be empty")
    return RetrievalVocabularies(user_ids=user_ids, movie_ids=movie_ids)


def build_evaluation_users(
    validation_context_path: Path,
    train_ratings_path: Path,
    allowed_user_ids: np.ndarray,
    candidate_ids: np.ndarray,
    positive_rating: float,
    max_users: int,
    seed: int,
    parquet_batch_size: int,
) -> list[EvaluationUser]:
    """Build known-user future targets, excluding movies already seen in train."""
    allowed_users = set(int(user_id) for user_id in allowed_user_ids)
    candidates = set(int(movie_id) for movie_id in candidate_ids)
    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    validation_scanner = _parquet_dataset(validation_context_path).scanner(
        columns=["userId", "movieId", "rating", "is_new_user", "is_new_item"],
        batch_size=parquet_batch_size,
        use_threads=False,
    )
    for batch in validation_scanner.to_batches():
        user_ids = _column(batch, "userId")
        movie_ids = _column(batch, "movieId")
        ratings = _column(batch, "rating")
        new_users = _column(batch, "is_new_user").astype(bool, copy=False)
        new_items = _column(batch, "is_new_item").astype(bool, copy=False)
        keep = (ratings >= positive_rating) & ~new_users & ~new_items
        for user_id, movie_id in zip(user_ids[keep], movie_ids[keep], strict=True):
            if int(user_id) not in allowed_users:
                continue
            movie = int(movie_id)
            if movie in candidates:
                relevant_by_user[int(user_id)].add(movie)

    if not relevant_by_user:
        raise ValueError("No known-user positive validation targets were found")

    eligible_users = np.array(sorted(relevant_by_user), dtype=np.int64)
    seen_by_user: dict[int, set[int]] = defaultdict(set)
    train_scanner = _parquet_dataset(train_ratings_path).scanner(
        columns=["userId", "movieId"],
        batch_size=parquet_batch_size,
        use_threads=False,
    )
    for batch in train_scanner.to_batches():
        user_ids = _column(batch, "userId")
        keep = np.isin(user_ids, eligible_users)
        if not keep.any():
            continue
        movie_ids = _column(batch, "movieId")
        for user_id, movie_id in zip(user_ids[keep], movie_ids[keep], strict=True):
            seen_by_user[int(user_id)].add(int(movie_id))

    ordered_users = eligible_users.tolist()
    random.Random(seed).shuffle(ordered_users)
    evaluation_users = []
    for user_id in ordered_users:
        seen = seen_by_user[user_id]
        relevant = relevant_by_user[user_id] - seen
        if not relevant:
            continue
        evaluation_users.append(
            EvaluationUser(
                user_id=user_id,
                relevant_items=frozenset(relevant),
                seen_items=frozenset(seen),
            )
        )
        if len(evaluation_users) >= max_users:
            break

    if not evaluation_users:
        raise ValueError("No validation users retained unseen positive targets")
    return evaluation_users


def load_popularity_order(
    item_features_path: Path, candidate_ids: np.ndarray
) -> list[int]:
    """Return candidate IDs in deterministic training-popularity order."""
    candidates = set(int(movie_id) for movie_id in candidate_ids)
    ranked: list[tuple[int, int]] = []
    scanner = _parquet_dataset(item_features_path).scanner(
        columns=["movieId", "popularity_rank"], use_threads=False
    )
    for batch in scanner.to_batches():
        movie_ids = _column(batch, "movieId")
        ranks = batch.column(
            batch.schema.get_field_index("popularity_rank")
        ).to_pylist()
        for movie_id, rank in zip(movie_ids, ranks, strict=True):
            movie = int(movie_id)
            if movie in candidates and rank is not None:
                ranked.append((int(rank), movie))
    return [movie_id for _, movie_id in sorted(ranked)]
