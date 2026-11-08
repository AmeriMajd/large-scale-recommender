"""Idempotent retraining stages and lightweight artifact validation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    make_target: str | None
    dependencies: tuple[str, ...]
    required_outputs: tuple[str, ...]


STAGES = (
    PipelineStage(
        "validate_raw",
        "download",
        (),
        ("data/raw/ml-25m/ratings.csv", "data/raw/ml-25m/movies.csv"),
    ),
    PipelineStage(
        "prepare_data",
        "prepare",
        ("validate_raw",),
        (
            "data/processed/ml-25m/ratings_train/_SUCCESS",
            "data/processed/ml-25m/ratings_validation/_SUCCESS",
            "data/processed/ml-25m/ratings_test/_SUCCESS",
        ),
    ),
    PipelineStage(
        "build_features",
        "features",
        ("prepare_data",),
        (
            "data/processed/ml-25m/features/user_features/_SUCCESS",
            "data/processed/ml-25m/features/item_features/_SUCCESS",
        ),
    ),
    PipelineStage(
        "prepare_retrieval",
        "prepare-retrieval",
        ("build_features",),
        ("data/processed/ml-25m/retrieval/train_sample/_SUCCESS",),
    ),
    PipelineStage(
        "train_retrieval",
        "train-retrieval",
        ("prepare_retrieval",),
        ("artifacts/models/retrieval/latest.json",),
    ),
    PipelineStage(
        "build_index",
        "build-index",
        ("train_retrieval",),
        ("artifacts/indexes/latest.json",),
    ),
    PipelineStage(
        "prepare_ranking",
        "prepare-ranking",
        ("build_index", "build_features"),
        (
            "data/processed/ml-25m/ranking/training_examples/_SUCCESS",
            "data/processed/ml-25m/ranking/validation_examples/_SUCCESS",
        ),
    ),
    PipelineStage(
        "train_ranking",
        "train-ranking",
        ("prepare_ranking",),
        ("artifacts/models/ranking/latest.json",),
    ),
    PipelineStage(
        "export_triton",
        "export-triton",
        ("train_retrieval", "train_ranking"),
        (
            "triton-models/user_tower/1/model.savedmodel/saved_model.pb",
            "triton-models/wide_and_deep/1/model.savedmodel/saved_model.pb",
            "triton-models/dcn/1/model.savedmodel/saved_model.pb",
        ),
    ),
    PipelineStage(
        "validate_serving",
        None,
        ("export_triton", "build_index"),
        ("triton-models/manifest.json", "artifacts/indexes/latest.json"),
    ),
)

STAGE_BY_ID = {stage.stage_id: stage for stage in STAGES}


def validate_graph() -> None:
    """Reject missing dependencies and cycles before Airflow parses the DAG."""
    if len(STAGE_BY_ID) != len(STAGES):
        raise ValueError("pipeline stage IDs must be unique")
    for stage in STAGES:
        missing = set(stage.dependencies) - set(STAGE_BY_ID)
        if missing:
            raise ValueError(f"{stage.stage_id} has missing dependencies: {missing}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise ValueError(f"pipeline contains a cycle at {stage_id}")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for dependency in STAGE_BY_ID[stage_id].dependencies:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage in STAGES:
        visit(stage.stage_id)


def validate_outputs(stage: PipelineStage, project_root: Path) -> list[Path]:
    missing = [
        project_root / relative
        for relative in stage.required_outputs
        if not (project_root / relative).is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"{stage.stage_id} missing outputs: {formatted}")
    return [project_root / relative for relative in stage.required_outputs]


def run_stage(stage_id: str, mode: str, project_root: Path) -> list[Path]:
    """Run a full idempotent Make target or validate existing stage outputs."""
    validate_graph()
    if stage_id not in STAGE_BY_ID:
        raise ValueError(f"unknown pipeline stage: {stage_id}")
    if mode not in {"validate", "full"}:
        raise ValueError("pipeline mode must be validate or full")
    stage = STAGE_BY_ID[stage_id]
    if mode == "full" and stage.make_target is not None:
        subprocess.run(
            ["make", stage.make_target, "CONFIG=configs/full.toml"],
            cwd=project_root,
            check=True,
        )
    return validate_outputs(stage, project_root)


validate_graph()
