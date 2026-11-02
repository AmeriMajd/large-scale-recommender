# Large-Scale Personalized Recommendation Engine

A production-style, two-stage movie recommendation system built with MovieLens
25M. A TensorFlow Recommenders two-tower model retrieves a few hundred
candidates, then a deep ranking model re-scores them to produce the final
recommendations.

This repository is an internship project for the AI / Machine Learning
specialization track. The goal is not only to train a model, but to demonstrate
the complete lifecycle: data processing, training, evaluation, approximate
nearest-neighbor retrieval, serving, caching, orchestration, and experimentation.

## Current status

Project scaffold and implementation plan created. The first engineering
milestone is a leak-free temporal data split and a popularity baseline.

## System at a glance

1. Apache Spark cleans MovieLens 25M and creates time-based data splits.
2. Spark builds user, item, genre, and popularity features.
3. TensorFlow Recommenders trains user and item towers.
4. FAISS searches item embeddings and returns about 200 candidates.
5. Wide & Deep and Deep & Cross Network rankers re-score the candidates.
6. A serving API applies cold-start fallbacks and caches results in Redis.
7. NVIDIA Triton serves the TensorFlow models.
8. Apache Airflow schedules the training and indexing pipeline.
9. Deterministic A/B assignment compares the two ranking models.

The complete design, schedule, evaluation plan, and learning roadmap are in
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## Repository layout

```text
.
├── app/                 # Interactive recommendation demo
├── configs/             # Pipeline and model configuration
├── dags/                # Airflow retraining DAGs
├── data/                # Local raw/processed data; contents are not committed
├── docs/                # Architecture, API, and project documentation
├── notebooks/           # Exploration only; production logic belongs in src/
├── scripts/             # Reproducible command-line entry points
├── src/recommender/
│   ├── data/            # Spark ingestion, cleaning, and temporal splitting
│   ├── features/        # User and item feature pipelines
│   ├── retrieval/       # TensorFlow Recommenders two-tower model
│   ├── ranking/         # Wide & Deep and DCN rankers
│   ├── indexing/        # FAISS index creation and search
│   ├── evaluation/      # Offline quality and latency metrics
│   └── serving/         # API, Redis caching, and A/B assignment
├── tests/               # Unit and integration tests
└── triton-models/       # Triton model repository definitions
```

## Planned technology

- Python 3.11
- Apache Spark / PySpark
- TensorFlow and TensorFlow Recommenders
- FAISS
- Redis
- NVIDIA Triton Inference Server
- Apache Airflow
- Docker Compose

Python 3.11 is intentional: the host currently has Python 3.14, while the
documented TensorFlow binary support range is Python 3.9–3.12. The project will
use an isolated environment or container rather than modify the system Python.

## Dataset

The project uses [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/),
which contains 25,000,095 ratings and 1,093,360 tag applications across 62,423
movies from 162,541 users. Raw data must not be committed to Git.

## Project principles

- Split interactions chronologically, never randomly.
- Fit features using training data only to prevent leakage.
- Compare learned models against simple baselines.
- Develop on a small sample, then run the final pipeline on the full dataset.
- Keep notebooks exploratory and reusable logic tested in `src/`.
- Label simulated or demo A/B results honestly; MovieLens has no impression log.
- Record reproducible configuration, seeds, data cutoffs, metrics, and latency.

## Deliverables

- Working demo, deployed or screen-recorded
- GitHub repository with clean commits and setup instructions
- Architecture and data/API documentation
- Test results, accuracy metrics, and performance measurements
- Ten-minute final presentation
- Individual reflection describing learning and ownership

## References

- [MovieLens 25M README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html)
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)
- [TensorFlow Recommenders DCN tutorial](https://www.tensorflow.org/recommenders/examples/dcn)
- [FAISS getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [Triton backend documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/README.html)
- [Apache Airflow documentation](https://airflow.apache.org/docs/)

