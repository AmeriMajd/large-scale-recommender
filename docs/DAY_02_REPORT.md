# Day 2 Report: Leakage-Safe Spark Feature Tables

## Outcome

Day 2 turns the chronological MovieLens splits into model-ready Spark feature
tables. The pipeline now produces one row per training user, one row per catalog
movie, normalized user–genre preferences, and validation/test context tables
that identify cold-start cases.

All learned statistics stop at 22 June 2016 05:34:47 UTC. Future interactions
can be labeled for evaluation, but they cannot change a training count, average,
popularity rank, recency value, or preference share.

## Results

### Persisted feature snapshot

| Measurement | Result |
|---|---:|
| Training interactions used | 19,986,521 |
| Training users | 137,823 |
| Catalog item rows | 62,423 |
| Items observed in training | 34,444 (55.18%) |
| Items unseen in training | 27,979 (44.82%) |
| Catalog items with release year after cutoff year | 5,398 (8.65%) |
| Items with unknown release year | 423 |
| User–genre rows | 2,255,079 |
| Recent-history window | 90 days |
| Positive-rating threshold | 4.0 |
| Popularity buckets | 10 |
| Total Parquet storage | 112 MiB |

The five table sizes are approximately 6.0 MiB for users, 3.4 MiB for items,
25 MiB for user–genre preferences, 35 MiB for validation context, and 44 MiB
for test context. A full local rerun on eight Spark workers took roughly two
minutes and 45 seconds. This was an operational observation, not a controlled
performance benchmark.

### Cold-start audit

| Measurement | Validation | Test |
|---|---:|---:|
| Interactions | 2,501,547 | 2,512,027 |
| Distinct users | 18,014 | 18,513 |
| Users unseen in training | 12,720 | 14,407 |
| Distinct items | 37,190 | 49,427 |
| Items unseen in training | 10,626 | 21,510 |
| Known-user + known-item interactions | 370,876 (14.83%) | 237,636 (9.46%) |
| Interactions involving a new user | 2,077,765 (83.06%) | 2,173,617 (86.53%) |
| Interactions involving a new item | 117,819 (4.71%) | 320,324 (12.75%) |

The last two rows overlap: `new_user_and_item` interactions belong to both. This
result is a strong reason to build popularity and content fallbacks later. An
ID-only personalized model cannot create a learned embedding for a user or item
that was absent from training.

The machine-readable audit is written locally to
`artifacts/reports/feature_summary.json`. Generated data and reports remain out
of Git because they are reproducible and much larger than source code.

## Concepts learned

### Feature snapshots have a time boundary

A feature is not just a value; it is a value *as of a particular time*. A movie
with 50,000 ratings today may have had only 10,000 at the model cutoff. Using
today's count to evaluate a historical model reveals the future.

Every learned output therefore contains `feature_cutoff_epoch`. This lets a
training or serving job verify that user, item, and preference features belong
to the same snapshot.

### Aggregations reduce many events into entity features

The ratings table has one row per interaction. `groupBy("userId")` converts it
into one row per user; `groupBy("movieId")` converts it into one row per item.
Counts describe activity or popularity, averages describe rating tendency, and
minimum/maximum timestamps describe lifetime and recency.

### Window functions preserve rows while adding group context

An aggregation collapses rows. A window function adds a result while retaining
the current row. We use a global ordered window for item rank, percentile, and
decile, and a user-partitioned window for the denominator of genre shares.

### Null and zero mean different things

An unseen catalog movie has zero training interactions because that count is
known. It keeps a null `mean_rating`, because no average can be estimated. A
missing release year is also null/unknown rather than zero or a fabricated
year.

### Cold start is an evaluation segment, not one single case

There are four states: both entities known, new user, new item, or both new.
Their solutions differ. New users need popularity/context signals; new items
need content features; both-new cases have no collaborative signal at all.

## Key code walkthrough

### 1. Validated feature configuration

[`base.toml`](../configs/base.toml) defines the 90-day recent window and ten
popularity buckets. [`config.py`](../src/recommender/config.py) loads them into a
frozen `FeatureConfig` and rejects non-positive windows or fewer than two
buckets. Configuration keeps training, tests, and future orchestration aligned.

### 2. User aggregates

`build_user_features` in
[`build.py`](../src/recommender/features/build.py) groups training events by
`userId`. It computes interaction counts, distinct movies, rating distribution,
positive rate, first/last timestamps, recent activity, active span, and days
since the latest event.

The recent threshold is `cutoff - 90 days`, not the machine's current time.
This is necessary for reproducible historical training.

### 3. Item aggregates plus full-catalog metadata

`build_item_features` first aggregates only training events. A deterministic
global window orders items by interaction count, mean rating, and movie ID. It
then left-joins those values onto the complete movie catalog.

The left join matters: an inner join would silently delete new items. Unseen
items instead remain present with `is_seen_in_train = false`, zero counts, and
null rating statistics. `is_future_release_at_cutoff` marks obvious future
releases that must be excluded from a historical candidate set. Because
MovieLens supplies a release year rather than a full date, the flag cannot
resolve whether a 2016 movie was available before the June cutoff.

### 4. User–genre preferences

`build_user_genre_features` joins movie metadata, explodes each `genre_list`,
and aggregates by `(userId, genre)`. A user-partitioned window calculates total
genre events so each row can expose historical and recent preference shares.

Because one movie can have several genres, these are genre-event shares rather
than shares of distinct movies. This definition is documented rather than left
implicit.

### 5. Cold-start context

`mark_cold_start` builds distinct known-user and known-item sets from training,
then left-joins them onto a later interaction table. Null membership markers
become `is_new_user` and `is_new_item`; a conditional expression assigns one of
the four readable segment labels.

This transformation uses future rows only as objects to classify. It does not
aggregate future behavior back into a training feature.

### 6. Persistence and auditing

`run_feature_pipeline` caches the repeatedly used training and movie tables,
materializes the cache, builds all five outputs, writes Parquet, and unpersists
the inputs in a `finally` block. It reloads the generated files before profiling
them, so the audit verifies persisted outputs rather than only an in-memory
plan.

[`FEATURE_CONTRACT.md`](FEATURE_CONTRACT.md) is the consumer-facing contract for
paths, semantics, null behavior, time rules, and leakage guarantees.

## Complete debugging log

| # | Symptom or surprise | Root cause | Fix or decision | Prevention or test |
|---:|---|---|---|---|
| 1 | Ruff still reported two unsorted import blocks after formatting | `ruff format` formats layout but import sorting is a lint fix | Ran `ruff check --fix` on the affected files | `make lint` remains a completion gate |
| 2 | Spark repeatedly warned that a window had no partition and moved data to one partition | Exact global popularity rank, percentile, and `ntile` require a total order | Kept the exact window because it runs after aggregation over only 34,444 training items | Documented the tradeoff; at a million-item catalog, benchmark or replace bucket creation with approximate quantiles |
| 3 | The first audit keys said “cold-start counts” but did not say whether they counted users or interactions | The values came from `groupBy(cold_start_type).count()`, so they counted interaction rows | Replaced the ambiguous structure with distinct users/items plus `interactions_by_type`, then reran the full pipeline | Report field names now encode the unit |
| 4 | More than 80% of future interactions involved users unseen at the training cutoff | A global chronological split includes users who join the dataset after June 2016; this is not data corruption | Kept the honest temporal split and made cold-start segments first-class outputs | Later model evaluation must report known-user quality separately and exercise fallbacks for new users |
| 5 | Spark printed `NativeCodeLoader` because native Hadoop binaries were unavailable | The local development environment uses Spark's built-in Java implementations | No code change; the full job and tests completed correctly | Treat this as an environment warning unless storage/codec benchmarks show a problem |
| 6 | The agent execution environment required permission for Spark tests and jobs | Local Spark uses loopback sockets between Python and its JVM workers | Ran the approved local Spark commands | This restriction is specific to the managed agent sandbox, not the repository setup |
| 7 | One documentation patch initially failed to match its target text | Ruff had reformatted the relevant source lines after the patch context was prepared | Re-read the small section and applied a narrower patch | Inspect current context before subsequent edits; no generated data was affected |

There were no correctness failures during either full-data feature run. The
important unexpected result was the cold-start distribution, which changed our
understanding of how evaluation must be segmented.

## Tests and quality checks

- 14 automated tests pass in 21.92 seconds on the final local run.
- Ruff passes and `git diff --check` reports no whitespace errors.
- Two new integration tests run a real local Spark session.
- Exact-value checks cover user counts, averages, positive rate, 90-day
  recency, activity span, item rank, unseen-item null/zero behavior, future and
  missing release years, genre shares, and all four cold-start labels.
- A deliberate leakage test adds a future-only user, future-only item, and an
  extreme future rating. It verifies that the user is absent from learned user
  features, the item keeps zero training interactions, and the training user
  average cannot change.
- The full output audit reconciles validation/test interaction totals with the
  Day 1 split counts.

Not yet tested: behavior on a distributed multi-machine Spark cluster or a
catalog large enough for the exact global item window to become a bottleneck.

## Learning resources, in order

1. [scikit-learn: data leakage and correct preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
   — 20 minutes. Focus on the rule that transformations are learned from train
   only, then applied to later data. Map “fit” to our Spark aggregations.
2. [PySpark 4.0.1 `DataFrame.groupBy`](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.groupBy.html)
   — 20 minutes. Run the examples, then identify the grouping key and output
   grain of each Day 2 builder.
3. [PySpark 4.0.1 `DataFrame.join`](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html)
   — 20 minutes. Compare inner and left joins. Explain why item features and
   cold-start membership require left joins.
4. [Spark 4.0.1 window functions](https://spark.apache.org/docs/4.0.1/sql-ref-syntax-qry-select-window.html)
   — 30 minutes. Focus on `PARTITION BY`, `ORDER BY`, `row_number`,
   `percent_rank`, `ntile`, and aggregate functions over windows.
5. [Spark 4.0.1 SQL performance tuning](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html)
   — 25 minutes. Read caching, partitions, join strategy, and adaptive query
   execution. Connect caching to the repeated scans in `run_feature_pipeline`.
6. [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
   — 35-minute preview. Focus on query/candidate towers and embeddings. The Day
   2 tables will become inputs to this next modeling stage.

## Practice exercises

1. For a user with event times 120, 60, and 5 days before the cutoff, state
   their `recent_interaction_count` for a 90-day window. Then change the config
   to 30 days and rerun the tiny feature test.
2. Explain why an unseen item's `interaction_count` should be zero while its
   `mean_rating` should remain null.
3. A user rated one Action/Drama movie and two Comedy movies. Calculate the
   three genre-event shares by hand.
4. Add one synthetic validation row for a known user and unseen item. Predict
   every new cold-start column before running the test.
5. Replace the item catalog left join with an inner join in a temporary branch.
   Which rows disappear, and why would that hide a production problem?
6. Explain why `datetime.now()` would be wrong for recent-history features in a
   reproducible historical experiment.

### Exercise answer check

- Exercise 1: two events are within 90 days; only the five-day event is within
  30 days.
- Exercise 2: the count is observed to be zero at the cutoff, but an average of
  zero events is undefined.
- Exercise 3: Action is `1/4`, Drama is `1/4`, and Comedy is `2/4` because there
  are four genre events across three movie interactions.
- Exercise 4: `is_new_user = false`, `is_new_item = true`, and
  `cold_start_type = "new_item"`.
- Exercise 5: every catalog movie absent from training disappears, preventing
  measurement and handling of item cold start.
- Exercise 6: values would change depending on run date and could use time after
  the model cutoff.

## Reproduction commands

From the repository root, after Day 1 data preparation:

```bash
source .venv/bin/activate
make features
make test
make lint
```

Inspect the generated audit locally:

```bash
sed -n '1,260p' artifacts/reports/feature_summary.json
du -sh data/processed/ml-25m/features
```

## Next day

Day 3 will convert the training interactions and selected features into a
TensorFlow input pipeline and train the first TensorFlow Recommenders two-tower
retrieval model on a development sample before scaling it up.

The quality gate is: deterministic vocabulary handling, no future-only IDs in
training vocabularies, a saved user tower and item tower, Recall@K measured on
known-user validation cases, explicit cold-start routing, and a comparison
against the Day 1 popularity baseline.
