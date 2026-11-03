"""Train, export, and evaluate the Day 3 two-tower retrieval model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import tensorflow_recommenders as tfrs  # noqa: E402
import tf_keras as keras  # noqa: E402

from recommender.config import ProjectConfig, load_config  # noqa: E402
from recommender.retrieval.data import (  # noqa: E402
    build_evaluation_users,
    load_popularity_order,
    load_train_vocabularies,
    make_training_dataset,
    profile_training_sample,
)
from recommender.retrieval.evaluate import evaluate_retrieval  # noqa: E402
from recommender.retrieval.fallbacks import choose_retrieval_strategy  # noqa: E402
from recommender.retrieval.model import build_model, export_tower  # noqa: E402


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_cutoff(processed_dir: Path) -> int:
    summary_path = processed_dir / "split_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Split summary does not exist: {summary_path}")
    return int(
        json.loads(summary_path.read_text(encoding="utf-8"))["train_cutoff_epoch"]
    )


def _routing_manifest() -> dict[str, str]:
    return {
        "known_user_and_item": choose_retrieval_strategy(
            is_new_user=False, is_new_item=False
        ).value,
        "new_user": choose_retrieval_strategy(
            is_new_user=True, is_new_item=False
        ).value,
        "new_item": choose_retrieval_strategy(
            is_new_user=False, is_new_item=True
        ).value,
        "new_user_and_item": choose_retrieval_strategy(
            is_new_user=True, is_new_item=True
        ).value,
    }


def run_training(config: ProjectConfig) -> dict:
    """Run the bounded CPU development experiment and persist its artifacts."""
    cutoff = _load_cutoff(config.data.processed_dir)
    feature_dir = config.data.processed_dir / "features"
    full_train_path = config.data.processed_dir / "ratings_train"
    train_path = config.data.processed_dir / "retrieval" / "train_sample"
    validation_context_path = feature_dir / "validation_context"
    vocabularies = load_train_vocabularies(feature_dir)

    retrieval = config.retrieval
    sample_profile = profile_training_sample(
        train_path,
        config.baseline.positive_rating,
        retrieval.parquet_batch_size,
        retrieval.max_train_interactions,
        retrieval.random_seed,
    )
    if sample_profile["interactions"] == 0:
        raise ValueError("Retrieval training sample is empty")

    dataset = make_training_dataset(
        train_path,
        config.baseline.positive_rating,
        retrieval.parquet_batch_size,
        retrieval.max_train_interactions,
        retrieval.batch_size,
        retrieval.shuffle_buffer,
        retrieval.random_seed,
    )
    model = build_model(
        vocabularies.user_ids,
        vocabularies.movie_ids,
        retrieval.embedding_dimension,
        retrieval.learning_rate,
        retrieval.random_seed,
    )
    steps_per_epoch = math.ceil(sample_profile["interactions"] / retrieval.batch_size)
    started = time.perf_counter()
    history = model.fit(
        dataset,
        epochs=retrieval.epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
    )
    training_seconds = time.perf_counter() - started

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"cutoff-{cutoff}-seed-{retrieval.random_seed}-{timestamp}"
    run_dir = retrieval.model_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    np.save(run_dir / "user_ids.npy", vocabularies.user_ids)
    np.save(run_dir / "candidate_movie_ids.npy", vocabularies.movie_ids)
    export_tower(model.user_tower, run_dir / "user_tower", "user_id")
    export_tower(model.item_tower, run_dir / "item_tower", "movie_id")

    evaluation_users = build_evaluation_users(
        validation_context_path,
        full_train_path,
        vocabularies.user_ids,
        vocabularies.movie_ids,
        config.baseline.positive_rating,
        retrieval.max_validation_users,
        retrieval.random_seed,
        retrieval.parquet_batch_size,
    )
    popularity_order = load_popularity_order(
        feature_dir / "item_features", vocabularies.movie_ids
    )
    evaluation, item_embeddings = evaluate_retrieval(
        model.user_tower,
        model.item_tower,
        vocabularies.movie_ids,
        evaluation_users,
        popularity_order,
        retrieval.evaluation_ks,
        retrieval.batch_size,
    )
    np.save(run_dir / "item_embeddings.npy", item_embeddings)

    history_values = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }
    report = {
        "run_id": run_id,
        "feature_cutoff_epoch": cutoff,
        "feature_cutoff_utc": datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
        "framework_versions": {
            "tensorflow": tf.__version__,
            "tensorflow_recommenders": tfrs.__version__,
            "tf_keras": keras.__version__,
        },
        "device": "CPU",
        "training": {
            "positive_rating_threshold": config.baseline.positive_rating,
            "embedding_dimension": retrieval.embedding_dimension,
            "batch_size": retrieval.batch_size,
            "epochs": retrieval.epochs,
            "learning_rate": retrieval.learning_rate,
            "steps_per_epoch": steps_per_epoch,
            "sample": sample_profile,
            "seconds": training_seconds,
            "history": history_values,
        },
        "vocabularies": {
            "training_users": len(vocabularies.user_ids),
            "candidate_items": len(vocabularies.movie_ids),
            "user_ids_sha256": _array_sha256(vocabularies.user_ids),
            "movie_ids_sha256": _array_sha256(vocabularies.movie_ids),
            "unknown_id_buckets": 1,
            "source": "training-only Day 2 feature snapshot",
        },
        "evaluation": evaluation,
        "cold_start_routing": _routing_manifest(),
        "artifacts": {
            "run_dir": str(run_dir),
            "user_tower": str(run_dir / "user_tower"),
            "item_tower": str(run_dir / "item_tower"),
            "candidate_movie_ids": str(run_dir / "candidate_movie_ids.npy"),
            "item_embeddings": str(run_dir / "item_embeddings.npy"),
        },
    }
    (run_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    retrieval.model_dir.mkdir(parents=True, exist_ok=True)
    (retrieval.model_dir / "latest.json").write_text(
        json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.data.reports_dir / "retrieval_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--max-train-interactions", type=int)
    parser.add_argument("--max-validation-users", type=int)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        name: value
        for name, value in {
            "max_train_interactions": args.max_train_interactions,
            "max_validation_users": args.max_validation_users,
            "epochs": args.epochs,
        }.items()
        if value is not None
    }
    if overrides:
        retrieval = replace(config.retrieval, **overrides)
        retrieval.validate()
        config = replace(config, retrieval=retrieval)
    report = run_training(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
