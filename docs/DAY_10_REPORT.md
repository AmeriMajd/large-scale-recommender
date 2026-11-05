# Day 10 Report: Airflow Retraining Orchestration

## Outcome

Day 10 turns the existing commands into one dependency-aware retraining DAG.
Ten stages cover raw-data validation, temporal preparation, features, retrieval
sampling/training, FAISS, ranking preparation/training, Triton export, and final
serving validation. The graph runs weekly with catchup disabled, one active run,
and one retry per task.

Two execution modes solve different problems. `validate` checks the existing
25M-data artifact chain quickly and is intended for the local DAG
demonstration. `full` executes every stage's established Make target before
checking outputs. The local Python validator and the pinned Airflow 3.3.2
container both successfully checked all ten stages and fourteen declared
output artifacts.

## Concepts learned

Airflow orchestrates work; it should not contain the ML implementation. Every
task delegates to a versioned project command, which remains runnable and
testable without a scheduler.

Retries require idempotency. A task must either produce a complete versioned
artifact and update its pointer, or fail before downstream work starts. Output
validation after each Make target makes that contract visible to Airflow.

DAG parsing must stay lightweight. The shared graph module imports only the
standard library; it does not load Spark, TensorFlow, FAISS, model files, or
data while the scheduler is discovering DAGs.

## Key code walkthrough

[`pipeline.py`](../src/recommender/orchestration/pipeline.py) defines immutable
stage IDs, Make targets, dependencies, and required outputs. It validates
unique IDs, missing edges, and cycles, then either executes or validates one
stage.

[`pipeline_stage.py`](../scripts/pipeline_stage.py) is the CLI boundary used by
Airflow. Environment variables select project root and default mode; explicit
arguments remain available for reproducible manual runs.

[`recommender_retraining.py`](../dags/recommender_retraining.py) creates one
`BashOperator` per shared stage and wires dependencies from the same
specification. It contains scheduling and retry policy, not model logic.

## Complete debugging log

| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| Importing Airflow into the main environment would risk dependency conflicts with TensorFlow/Spark | Airflow has a large, independently constrained dependency graph | Pinned the official Airflow 3.3.2 Python 3.11 image behind a Compose profile | Keep orchestration dependencies isolated from model dependencies |
| A full 25M retrain is too slow for a short DAG demonstration | Training success and orchestration correctness are different test scopes | Added `validate` mode that exercises the real DAG and artifact checks without retraining | Make demos fast without pretending validation is fresh training |
| A DAG-only task list would duplicate command and dependency knowledge | Pipeline commands already exist in the Makefile | Added a shared typed stage specification consumed by the CLI and DAG | One graph definition reduces drift |
| Downstream tasks could run after a command that exited zero but produced no artifact | Exit status alone does not prove the ML stage contract | Every stage declares and validates concrete files after execution | Add data/artifact self-checks to workflows |
| Dynamic DAG code is harder to inspect with string-only tests | Airflow is intentionally absent from the main virtual environment | Added graph unit tests and a reproducible `make airflow-test` command, then ran the DAG in the pinned Airflow container | Unit graph tests and scheduler integration tests cover different failures, so both need explicit evidence |

## Tests and evidence

- Graph validation proves ten unique acyclic stages.
- Dependency tests cover the feature/index join into ranking and both model
  branches into Triton export.
- Missing-output tests prove validation fails with a useful path.
- Local validation checked every declared artifact successfully.
- The full repository suite reports 57 passed, two service-gated skips, and one
  recorded transitive deprecation warning.
- `make airflow-test` pulled `apache/airflow:3.3.2-python3.11`, migrated the
  SQLite metadata database, executed all ten tasks in dependency order, and
  finished the DAG run successfully on 5 November 2026.

Validation mode proves orchestration and the current artifact chain. It does
not claim that a new multi-hour 25M retrain was performed during this milestone.

## Learning resources, in order

1. [Airflow core concepts](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/)
   — learn DAG runs, tasks, logical dates, retries, and scheduling.
2. [Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
   — focus on idempotent tasks, lightweight imports, self-checks, and DAG tests.
3. [Airflow Docker images](https://airflow.apache.org/docs/docker-stack/)
   — understand version/Python tags and dependency isolation.
4. [Airflow DAG testing](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/debug.html)
   — compare parser tests, task tests, and complete `dag.test()` runs.

## Practice exercises

1. Why must a retried training task write a new version rather than append to a
   shared partially written directory?
2. Add a data-quality gate that rejects an unexpectedly small training split.
3. Where would approval for production model promotion belong in this graph?
4. Explain why `validate` mode is useful but not evidence of fresh model
   quality.
5. Change the weekly schedule to a dataset-triggered workflow. What new
   idempotency key would you use?

## Reproduction

```bash
make validate-pipeline
make airflow-test
```

See [`AIRFLOW_CONTRACT.md`](AIRFLOW_CONTRACT.md) for graph and mode semantics.
