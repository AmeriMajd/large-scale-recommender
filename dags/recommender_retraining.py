"""Airflow DAG for the reproducible two-stage recommender pipeline."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator

from recommender.orchestration.pipeline import STAGES

PROJECT_ROOT = os.getenv("RECSYS_PROJECT_ROOT", "/opt/project")
MODE = os.getenv("RECSYS_DAG_MODE", "validate")

with DAG(
    dag_id="recommender_retraining",
    description="Validate or retrain retrieval, FAISS, ranking, and Triton exports",
    schedule="0 3 * * 1",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    params={"mode": MODE},
    tags=["recommendation", "ml", "retraining"],
) as dag:
    tasks = {
        stage.stage_id: BashOperator(
            task_id=stage.stage_id,
            bash_command=(
                f"python {PROJECT_ROOT}/scripts/pipeline_stage.py "
                f"--stage {stage.stage_id} --mode {{{{ params.mode }}}}"
            ),
            cwd=PROJECT_ROOT,
        )
        for stage in STAGES
    }
    for stage in STAGES:
        for dependency in stage.dependencies:
            tasks[dependency] >> tasks[stage.stage_id]
