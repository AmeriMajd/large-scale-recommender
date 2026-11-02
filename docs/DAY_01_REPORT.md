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

## Key code walkthrough

### 1. Configuration instead of hidden constants

[`configs/base.toml`](../configs/base.toml) contains paths, split fractions,
the positive-rating threshold, requested values of K, and local Spark settings.
[`config.py`](../src/recommender/config.py) converts the TOML dictionaries into
frozen dataclasses and validates important assumptions such as the split
fractions summing to 1.0.

This gives one source of truth. A future Airflow task, test, and command-line run
can all use the same configuration instead of copying values such as `0.8` or
`4.0` around the codebase.

### 2. Explicit schemas and fail-fast ingestion

[`schemas.py`](../src/recommender/data/schemas.py) describes every source column
with `StructType` and `StructField`. In `read_raw_tables`, Spark reads CSV with
`FAILFAST` parsing and those schemas instead of guessing types from a sample.

Schema inference is convenient for exploration but risky in a reproducible
pipeline. A malformed value could cause a whole numerical column to be treated
as text or allow a corrupted row to pass unnoticed.

### 3. DataFrame transformations are lazy

`clean_ratings` and `clean_movies` build transformation plans using `filter`,
`dropDuplicates`, `withColumn`, `split`, and `regexp_extract`. Spark does not
perform the full work when those lines execute. Work starts when an action such
as `approxQuantile`, `count`, `collect`, or `write` needs a result.

The cleaned ratings DataFrame is cached because quantile calculation, boundary
checks, row counts, and Parquet writes reuse it. After the outputs are written,
`unpersist()` releases the cached partitions.

### 4. Missing values are data, not necessarily errors

`clean_movies` parses a four-digit year only when a title ends with one. The
expression uses `when` so a title without a year receives `null`. It does not
invent a year and does not discard an otherwise valid movie.

`(no genres listed)` similarly becomes an empty string array rather than the
literal text being treated as a real genre.

### 5. The temporal split prevents future leakage

`temporal_split` asks Spark for the 80th and 90th timestamp quantiles. The
configured relative error is `0.001`, which is much cheaper than an exact global
sort while keeping the cutoff rank close to the requested value.

The comparisons are deliberately asymmetric:

```text
train:      timestamp <= train_cutoff
validation: timestamp > train_cutoff and timestamp <= validation_cutoff
test:       timestamp > validation_cutoff
```

Consequently, every timestamp belongs to exactly one split, and equal
timestamps stay together. `assert_temporal_order` independently calculates the
minimum and maximum of each split and fails if the ranges overlap.

### 6. Parquet is the processed-data boundary

`write_processed_data` writes movies and the three rating splits as Parquet.
Unlike CSV, Parquet stores typed, columnar data with its schema. Later jobs can
read only the columns they need and avoid repeating CSV parsing and cleaning.

`overwrite` makes a deliberate pipeline rerun replace a generated table. It is
safe here because raw data lives under a different path and processed data is
reproducible from source.

### 7. The baseline uses only information available in training

In [`popularity.py`](../src/recommender/baselines/popularity.py),
`popularity_ranking` groups only the training table by movie. Interaction count
is the main score; mean rating and movie ID provide deterministic tie-breaking.

Only the small ordered movie-ID catalog is collected to the driver. The 25
million interaction rows are never collected. Spark broadcasts the ordered
catalog once to workers, where `mapPartitions` evaluates many users without
shipping another catalog copy with every row.

For each test user:

- `relevant_items` contains future movies rated at least 4.0;
- `seen_items` contains movies observed for that user in training; and
- recommendations scan the popularity list while excluding `seen_items`.

The final Spark aggregation averages each user's metrics, producing a macro
average in which every user has equal weight.

### 8. Recall and NDCG answer different questions

[`metrics.py`](../src/recommender/evaluation/metrics.py) intentionally has no
Spark dependency, which makes its formulas easy to test.

For a recommendation list `R`, relevant set `T`, and cutoff `K`:

```text
Recall@K = |R[:K] intersection T| / |T|
```

Recall measures coverage but ignores order. Binary DCG discounts a relevant
item according to its rank:

```text
DCG@K = sum(1 / log2(rank + 1)) for relevant results
NDCG@K = DCG@K / ideal_DCG@K
```

The implementation uses zero-based Python indexes, so `rank + 2` inside
`log2` corresponds to the one-based formula. Dividing by ideal DCG normalizes
the score to the range from 0 to 1.

### 9. Runtime pinning is part of the application

[`spark.py`](../src/recommender/spark.py) pins UTC, memory, shuffle partitions,
and both Python executables. [`setup_java.py`](../scripts/setup_java.py)
downloads Java 17 into `.tools/`, safely extracts the archive, and records its
SHA-256 digest. The `Makefile` exports the local Java and Python paths for every
project command.

This is why `make prepare` behaves consistently even though the host defaults
are Python 3.14 and Java 25.

## Complete debugging log

| # | Symptom | Root cause | Fix | Protection against recurrence |
|---:|---|---|---|---|
| 1 | Initial dependency installation could not resolve PyPI | The execution sandbox blocks network access by default | Re-ran the explicitly approved setup with network access | `make setup` remains the single documented installation command; this restriction is specific to the agent sandbox |
| 2 | Ruff reported unsorted imports and lines over 88 characters | Newly written files had not yet been formatted | Applied Ruff's safe fixes and manually wrapped the remaining expressions | `make check` runs lint before the work is considered complete |
| 3 | Spark's Java gateway exited while inspecting network interfaces | The execution sandbox initially blocked Spark's local loopback sockets | Re-ran Spark tests with approved local socket access | The real integration test remains in the suite; normal local machines do not need this agent-specific approval |
| 4 | Spark failed with `UnsupportedOperationException: getSubject is not supported` | Host Java 25 removed behavior still used by Spark's bundled Hadoop version | Installed a project-local Eclipse Temurin 17 JDK and exported `JAVA_HOME` | `make setup-java`, the Makefile environment, and the local Java metadata make the runtime explicit |
| 5 | Spark workers reported Python 3.14 while the driver used 3.11 | Worker discovery found the system `python3` instead of the virtual environment | Set `PYSPARK_PYTHON`, `PYSPARK_DRIVER_PYTHON`, `spark.pyspark.python`, and `spark.pyspark.driver.python` | The Spark integration test runs Python work on workers and fails if versions diverge |
| 6 | The first full job produced many cache-to-disk warnings | Spark had a small default JVM heap and used all 16 CPU workers concurrently | Set a 6 GiB driver heap and limited local execution to eight workers | Settings live in `base.toml`; subsequent full runs completed without the spill warnings |
| 7 | The first full write failed with `CAST_INVALID_INPUT` for an empty release year | `regexp_extract` returns an empty string when a title has no trailing year, and Spark 4 uses strict numeric casts | Guarded the cast with `when(length(...) > 0, ...)`, producing `null` otherwise | Added a Spark regression test with both year and no-year movie titles |
| 8 | The failed full run left an incomplete processed-output attempt | Spark had begun writing generated tables before encountering the movie-year error | Fixed the defect and reran the reproducible pipeline with `overwrite` mode | Raw data is immutable and separate; all processed tables are regenerable and checked by final row counts |
| 9 | No NVIDIA GPU was available through the installed driver | The development host currently has no usable NVIDIA GPU | Kept Day 1 CPU-only; Spark, TensorFlow development, and FAISS CPU all remain viable | Later reports will separate CPU benchmarks from any optional cloud/GPU experiment |

The main debugging pattern was: read the first meaningful exception, identify
whether it came from code, data, runtime compatibility, or the execution
environment, make the smallest targeted fix, and add a test when the failure was
a project defect.

## Learning resources, in order

1. [Google's recommendation-system overview](https://developers.google.com/machine-learning/recommendation/overview/types)
   — 20 minutes. Focus on candidate generation, scoring, and re-ranking. This
   gives the mental model for the whole internship project.
2. [Official MovieLens 25M README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html)
   — 15 minutes. Read the rating, timestamp, movie, genre, and license sections.
   Compare each source definition with `DATA_CONTRACT.md`.
3. [Spark SQL and DataFrames guide](https://spark.apache.org/docs/4.0.1/sql-programming-guide.html)
   — 45 minutes. Learn transformations versus actions, lazy evaluation, and why
   DataFrames can be optimized.
4. [PySpark data types](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/data_types.html)
   — 15 minutes. Focus on `StructType`, `StructField`, `LongType`, `DoubleType`,
   `StringType`, and arrays.
5. [`DataFrame.approxQuantile` documentation](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.approxQuantile.html)
   — 15 minutes. Understand `probabilities`, `relativeError`, and the rank-error
   guarantee; do not worry about memorizing the underlying algorithm.
6. [Spark Parquet guide](https://spark.apache.org/docs/4.0.1/sql-data-sources-parquet.html)
   — 15 minutes. Focus on schema preservation and columnar storage.
7. [Spark SQL performance tuning](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html)
   — 20 minutes. Read caching and partition-tuning sections, then relate them to
   the memory-spill issue we observed.
8. [Spark `RankingMetrics`](https://spark.apache.org/docs/4.1.2/api/python/reference/api/pyspark.mllib.evaluation.RankingMetrics.html)
   — 25 minutes. Study the definitions of Recall@K and NDCG@K and compare them
   with our small framework-independent implementation.
9. [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
   — optional 30-minute preview. Stop after understanding user/item towers and
   top-K evaluation; we will implement this in a later day.

Do not try to read all Spark documentation. The goal is to connect these exact
sections to code you have already run.

## Practice exercises

1. For recommendations `[10, 20, 30, 40]` and relevant items `{20, 40, 50}`,
   calculate Recall@2 and Recall@4 by hand, then check your answer with
   `recall_at_k`.
2. Move item `20` from rank 2 to rank 1 and calculate how NDCG changes. Explain
   why Recall does not change.
3. Run `make prepare` and locate the four kinds of Spark actions in `prepare.py`
   that force lazy transformations to execute.
4. Explain why calculating movie popularity from train + validation + test would
   leak future information, even though popularity is not a neural-network
   feature.
5. Add a tiny test movie with malformed genres or a null title and predict which
   cleaning rule handles it before running the test.

### Exercise answer check

- Exercise 1: Recall@2 is `1/3`; Recall@4 is `2/3`.
- Exercise 2: Recall stays the same because the same relevant item is present;
  NDCG increases because the relevant item receives a smaller rank discount.
- Exercise 3: `approxQuantile`, the boundary aggregations ending in `first`,
  each `count`, and each Parquet `write` are actions. `collect` is also an action
  in the separate popularity job.
- Exercise 4: future interaction counts reveal which movies become popular
  after the model's training cutoff, information unavailable at prediction time.
- Exercise 5: a null title is filtered out. An unexpected non-null genre string
  currently becomes a one-element array; validating allowed genre values is a
  possible future data-quality improvement.

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
