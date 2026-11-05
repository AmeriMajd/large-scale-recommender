from dataclasses import replace
from pathlib import Path

import pytest

from recommender.orchestration import pipeline


def test_pipeline_graph_has_expected_dependencies() -> None:
    pipeline.validate_graph()

    assert len(pipeline.STAGES) == 10
    assert pipeline.STAGE_BY_ID["prepare_ranking"].dependencies == (
        "build_index",
        "build_features",
    )
    assert pipeline.STAGE_BY_ID["export_triton"].dependencies == (
        "train_retrieval",
        "train_ranking",
    )
    assert pipeline.STAGE_BY_ID["validate_serving"].make_target is None


def test_validate_mode_requires_declared_outputs(tmp_path: Path) -> None:
    stage = replace(
        pipeline.STAGE_BY_ID["validate_raw"],
        required_outputs=("one.txt", "nested/two.txt"),
    )
    (tmp_path / "one.txt").write_text("ok", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="nested/two.txt"):
        pipeline.validate_outputs(stage, tmp_path)

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/two.txt").write_text("ok", encoding="utf-8")
    assert len(pipeline.validate_outputs(stage, tmp_path)) == 2


def test_airflow_dag_declares_every_stage() -> None:
    dag_source = Path("dags/recommender_retraining.py").read_text(encoding="utf-8")

    assert 'dag_id="recommender_retraining"' in dag_source
    assert "catchup=False" in dag_source
    assert "max_active_runs=1" in dag_source
    assert "for stage in STAGES" in dag_source
