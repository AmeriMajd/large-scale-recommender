"""Train, compare, export, and report the Day 5 ranking models."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import tensorflow_recommenders as tfrs  # noqa: E402
import tf_keras as keras  # noqa: E402

from recommender.config import ProjectConfig, load_config  # noqa: E402
from recommender.ranking.data import (  # noqa: E402
    MODEL_FEATURES,
    RetrievalScoreLookup,
    make_ranking_dataset,
)
from recommender.ranking.evaluate import evaluate_candidate_order  # noqa: E402
from recommender.ranking.model import build_ranker, export_ranker  # noqa: E402


def _latest_retrieval_run(model_dir: Path) -> tuple[str, Path]:
    pointer = model_dir / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"retrieval pointer does not exist: {pointer}")
    value = json.loads(pointer.read_text(encoding="utf-8"))
    return str(value["run_id"]), Path(value["run_dir"])


def _load_validation_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    table = pads.dataset(str(path), format="parquet").to_table()

    def values(name: str, dtype) -> np.ndarray:
        return np.asarray(table[name].to_numpy(), dtype=dtype)

    features = {
        name: values(
            {"user_id": "userId", "movie_id": "movieId"}.get(name, name),
            np.int64 if name in {"user_id", "movie_id"} else np.float32,
        )
        for name in MODEL_FEATURES
    }
    metadata = {
        "labels": values("label", np.float32),
        "retrieval_rank": values("retrieval_rank", np.int64),
        "total_relevant_count": values("total_relevant_count", np.int64),
    }
    return features, metadata


def _history_values(history: keras.callbacks.History) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }


def _train_one(
    kind: str,
    config: ProjectConfig,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    validation_features: dict[str, np.ndarray],
    validation_metadata: dict,
    steps_per_epoch: int,
) -> tuple[keras.Model, dict]:
    ranking = config.ranking
    model = build_ranker(
        kind,
        user_ids,
        movie_ids,
        ranking.id_embedding_dimension,
        ranking.deep_layer_sizes,
        ranking.cross_layers,
        ranking.learning_rate,
        ranking.random_seed,
    )
    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=ranking.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=ranking.early_stopping_patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    candidate_metrics = {
        name: float(value)
        for name, value in model.evaluate(
            validation_dataset, return_dict=True, verbose=0
        ).items()
    }
    logits = model.predict(
        validation_features, batch_size=ranking.batch_size, verbose=0
    )
    list_metrics = evaluate_candidate_order(
        validation_features["user_id"],
        validation_metadata["labels"],
        validation_metadata["total_relevant_count"],
        np.asarray(logits),
        ranking.evaluation_ks,
    )
    return model, {
        "parameters": model.count_params(),
        "training_seconds": training_seconds,
        "history": _history_values(history),
        "candidate_metrics": candidate_metrics,
        "list_metrics": list_metrics,
    }


def run_training(config: ProjectConfig) -> dict:
    """Fit both rankers on identical data and persist a matched comparison."""
    ranking = config.ranking
    ranking_data_dir = config.data.processed_dir / "ranking"
    preparation_report_path = ranking_data_dir / "preparation_report.json"
    if not preparation_report_path.is_file():
        raise FileNotFoundError("run make prepare-ranking before training rankers")
    preparation = json.loads(preparation_report_path.read_text(encoding="utf-8"))
    retrieval_run_id, retrieval_run_dir = _latest_retrieval_run(
        config.retrieval.model_dir
    )
    user_ids = np.load(retrieval_run_dir / "user_ids.npy", allow_pickle=False)
    movie_ids = np.load(
        retrieval_run_dir / "candidate_movie_ids.npy", allow_pickle=False
    )
    scorer = RetrievalScoreLookup(retrieval_run_dir)
    training_path = Path(preparation["training"]["path"])
    validation_path = Path(preparation["validation"]["path"])
    steps_per_epoch = math.ceil(
        preparation["training"]["rows"] / ranking.batch_size
    )
    training_dataset = make_ranking_dataset(
        training_path,
        scorer,
        ranking.parquet_batch_size,
        ranking.batch_size,
        ranking.random_seed,
        repeat=True,
    )
    validation_dataset = make_ranking_dataset(
        validation_path,
        scorer,
        ranking.parquet_batch_size,
        ranking.batch_size,
        ranking.random_seed,
    )
    validation_features, validation_metadata = _load_validation_arrays(
        validation_path
    )
    baselines = {
        "retrieval": evaluate_candidate_order(
            validation_features["user_id"],
            validation_metadata["labels"],
            validation_metadata["total_relevant_count"],
            -validation_metadata["retrieval_rank"],
            ranking.evaluation_ks,
        ),
        "popularity": evaluate_candidate_order(
            validation_features["user_id"],
            validation_metadata["labels"],
            validation_metadata["total_relevant_count"],
            validation_features["item_popularity_score"],
            ranking.evaluation_ks,
        ),
    }

    models: dict[str, keras.Model] = {}
    results: dict[str, dict] = {}
    for kind in ("wide_and_deep", "dcn"):
        models[kind], results[kind] = _train_one(
            kind,
            config,
            user_ids,
            movie_ids,
            training_dataset,
            validation_dataset,
            validation_features,
            validation_metadata,
            steps_per_epoch,
        )

    primary_metric = f"ndcg_at_{ranking.evaluation_ks[0]}"
    selected_model = max(
        results, key=lambda name: results[name]["list_metrics"][primary_metric]
    )
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{retrieval_run_id}-ranking-seed-{ranking.random_seed}-{timestamp}"
    run_dir = ranking.model_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        export_ranker(model, run_dir / name)

    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_retrieval_run_id": retrieval_run_id,
        "framework_versions": {
            "tensorflow": tf.__version__,
            "tensorflow_recommenders": tfrs.__version__,
            "tf_keras": keras.__version__,
        },
        "training": {
            **preparation["training"],
            "batch_size": ranking.batch_size,
            "epochs": ranking.epochs,
            "early_stopping_patience": ranking.early_stopping_patience,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": ranking.learning_rate,
            "id_embedding_dimension": ranking.id_embedding_dimension,
            "deep_layer_sizes": ranking.deep_layer_sizes,
            "cross_layers": ranking.cross_layers,
        },
        "validation": preparation["validation"],
        "baselines": baselines,
        "models": results,
        "selection": {
            "model": selected_model,
            "metric": primary_metric,
            "value": results[selected_model]["list_metrics"][primary_metric],
        },
        "artifacts": {
            name: str(run_dir / name) for name in models
        },
    }
    (run_dir / "ranking_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    ranking.model_dir.mkdir(parents=True, exist_ok=True)
    (ranking.model_dir / "latest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "selected_model": selected_model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config.data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.data.reports_dir / "ranking_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_training(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
