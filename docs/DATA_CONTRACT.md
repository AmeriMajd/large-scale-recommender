# MovieLens 25M Data Contract

## Raw tables used on Day 1

### `ratings.csv`

| Column | Spark type | Constraint | Meaning |
|---|---|---|---|
| `userId` | long | Positive, non-null | Anonymous MovieLens user |
| `movieId` | long | Positive, non-null | MovieLens movie identifier |
| `rating` | double | 0.5 through 5.0 | Explicit rating in half-star steps |
| `timestamp` | long | Positive, non-null | Seconds since the Unix epoch in UTC |

Exact duplicate rows are removed. The pipeline adds `event_time`, a UTC Spark
timestamp derived from `timestamp`.

### `movies.csv`

| Column | Spark type | Constraint | Meaning |
|---|---|---|---|
| `movieId` | long | Positive, unique, non-null | MovieLens movie identifier |
| `title` | string | Non-null | Display title, often ending in a year |
| `genres` | string | Non-null | Pipe-delimited genres |

The pipeline adds `genre_list` and a nullable `release_year`. The literal value
`(no genres listed)` becomes an empty genre array.

## Temporal split contract

The split uses two global timestamp quantiles calculated from cleaned ratings:

- training: timestamps at or before the 80% cutoff;
- validation: timestamps after the 80% cutoff and at or before the 90% cutoff;
- test: timestamps after the 90% cutoff.

Events with identical timestamps remain in the same split. The pipeline asserts:

```text
max(train.timestamp) < min(validation.timestamp)
max(validation.timestamp) < min(test.timestamp)
```

All learned statistics—including item popularity—must use the training split
only. Validation and test represent future data.

## Processed outputs

The Spark pipeline writes compressed Parquet tables under
`data/processed/ml-25m/`:

- `movies/`
- `ratings_train/`
- `ratings_validation/`
- `ratings_test/`
- `split_summary.json`

Raw and processed datasets are local artifacts and are excluded from Git.

