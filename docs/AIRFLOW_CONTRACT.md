# Airflow Retraining Contract

## Schedule and modes

The `recommender_retraining` DAG is scheduled at 03:00 UTC each Monday, has
catchup disabled, permits only one active run, and retries a failed task once
after five minutes.

- `validate` mode checks the complete existing artifact chain without spending
  hours retraining. This is the default for local demonstration and CI.
- `full` mode executes the corresponding idempotent Make target, then validates
  the declared outputs before allowing downstream tasks to run.

The mode is a DAG parameter initialized by `RECSYS_DAG_MODE`. A production
deployment should separate validation, training, and promotion permissions and
write artifacts to immutable object storage.

## Graph

```text
validate_raw → prepare_data → build_features → prepare_retrieval
                                              ↓
                                      train_retrieval → build_index
                                              ↓              ↓
                                      prepare_ranking ←───────┘
                                              ↓
                                        train_ranking
                                              ↓
retrieval + ranking ────────────────→ export_triton
                                              ↓
index + Triton export ─────────────→ validate_serving
```

The code-defined specification contains ten stages. Airflow creates one
`BashOperator` per stage, while the shared Python runner owns command selection
and output validation. This keeps the graph testable without importing Airflow
into the TensorFlow/Spark environment.

## Idempotency and promotion

Each full task calls an existing reproducible Make target. Tasks do not update
the next-stage pointer until their underlying pipeline succeeds. Every task
then checks concrete `_SUCCESS`, `latest.json`, SavedModel, FAISS, or manifest
outputs. Airflow retries therefore repeat a complete stage rather than trusting
partial files.

`make airflow-test` migrates an ignored local SQLite metadata database and runs
the entire DAG for one logical date in validate mode. This demonstrates graph
parsing, dependency ordering, templating, task execution, and artifact checks;
it is not evidence of a multi-node production Airflow deployment.
