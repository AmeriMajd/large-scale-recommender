# Day 1 Report: Environment, Temporal Split, and Baseline

## Outcome

Day 1 established a reproducible development environment and a complete
baseline pipeline over MovieLens 25M. The project can now download the dataset,
validate and clean it with Spark, write chronological Parquet splits, and score
a train-only popularity recommender against future positive interactions.

## Reproducible environment

The host versions were too new for parts of the required ML stack:

- host Python: 3.14;
- project Python: 3.11;
- host Java: 25; and
- project Java: Eclipse Temurin 17.

Spark initially failed because its bundled Hadoop library called a Java
authentication API that is unsupported in Java 25. After Java 17 was isolated,
Spark workers initially found the host Python 3.14 while the driver used Python
3.11. Both driver and worker interpreters are now explicitly pinned to the
project virtual environment.

This is an important distributed-systems lesson: every worker must use a
runtime compatible with the driver, even when the "cluster" is one laptop.

`make setup` installs both isolated runtimes without changing the system Python
or Java.

## Dataset provenance

| Field | Value |
|---|---|
| Source | GroupLens MovieLens 25M |
| Download date | 2 November 2026 UTC |
| Archive SHA-256 | `8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa` |
| Extracted raw size | Approximately 1.1 GiB |
| Processed Parquet size | Approximately 490 MiB |

The raw and processed data are excluded from Git. The download script writes a
local provenance record beside the source files.

## Temporal split

The split is global and chronological. Equal timestamps remain in the same
split so one instant cannot leak across two partitions.

| Split | Time rule | Rows | Share |
|---|---|---:|---:|
| Train | At or before 22 June 2016 05:34:47 UTC | 19,986,521 | 79.95% |
| Validation | After train and at or before 31 December 2017 21:39:11 UTC | 2,501,547 | 10.01% |
| Test | After the validation cutoff | 2,512,027 | 10.05% |
| Total | All cleaned ratings | 25,000,095 | 100% |

The row shares are close to, but not exactly, 80/10/10 because interactions
with the same timestamp stay together.

The global strategy answers a realistic question: if the system were trained in
June 2016 and later updated at the end of 2017, how well would it predict
subsequent behavior? A random split would let the model learn from the future.

## Baseline definition

The popularity baseline:

1. counts interactions for each movie using training data only;
2. orders movies by interaction count, mean rating, and movie ID;
3. removes movies each user already rated during training;
4. treats test ratings of 4.0 or higher as relevant future items; and
5. recommends the first 10 or 20 remaining popular movies.

This model is not personalized beyond filtering already-seen movies. Its value
is as a simple quality floor: the neural models must improve on it.

## Baseline results

| Metric | Result |
|---|---:|
| Evaluated users with at least one future positive | 18,344 |
| Future positive events | 1,245,827 |
| Catalog items observed in training | 34,444 |
| Users receiving fewer than 20 recommendations | 0 |
| Recall@10 | 0.045482 |
| NDCG@10 | 0.240805 |
| Recall@20 | 0.073187 |
| NDCG@20 | 0.212729 |

Metrics are macro-averaged: every evaluated user contributes equally, rather
than highly active users dominating the result.

### Interpretation

- Recall@10 of 0.0455 means the top 10 popularity recommendations recover, on
  average, about 4.55% of each user's relevant future movies.
- Recall improves to 7.32% at 20 because a longer list has more opportunities to
  contain a future positive.
- NDCG also considers position. A relevant item near rank 1 is more valuable
  than the same item near rank 10.
- NDCG@20 can be lower than NDCG@10. Its ideal normalization grows when the list
  grows, so weak additional ranks may not compensate for the larger ideal gain.
- Only 34,444 of the full 62,423-movie metadata catalog appeared during the
  training period. Later movies are genuine item cold-start cases and must not
  silently receive learned ID embeddings.

## Tests and quality checks

- 10 automated tests pass.
- Ruff linting passes.
- The Spark integration tests execute a real local Spark job.
- Tests cover configuration validation, split ordering, missing release years,
  Recall/NDCG calculations, seen-item filtering, and short catalogs.
- The full split preserves all 25,000,095 ratings.

The full-data run found a metadata edge case that the original fixture missed:
some movie titles have no parenthesized release year. The cleaning pipeline now
stores a null year instead of attempting to cast an empty string, and a
regression test protects the behavior.

## Reproduction commands

```bash
make setup
make download
make prepare
make baseline
make check
```

Generated outputs:

```text
data/raw/ml-25m/download_metadata.json
data/processed/ml-25m/split_summary.json
artifacts/reports/popularity_baseline.json
```

The generated outputs are local and ignored by Git; stable headline results are
recorded in this report.

## What Day 2 must add

Day 2 will build leakage-safe feature tables from the training window:

- user interaction count and rating statistics;
- historical and recent genre preferences;
- item count, mean rating, and popularity buckets;
- movie release year/age and genres; and
- validation/test transformations that reuse training-derived statistics.

The next quality gate is explicit tests proving that validation or test events
cannot change a training-derived feature.

