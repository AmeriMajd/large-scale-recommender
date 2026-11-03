# Feature Contract

This document defines the Day 2 Spark feature snapshot. Every learned statistic
is fitted from `ratings_train` only, whose latest timestamp is
`feature_cutoff_epoch`. Validation and test interactions are never inputs to a
user preference, item popularity, rating average, or recency calculation.

## Snapshot rules

- The default recent window is the 90 days ending at the training cutoff.
- A positive interaction is a rating greater than or equal to `4.0`.
- Null means “unknown” or “not estimable”; zero means a measured count of zero.
- Full-catalog metadata is retained to identify item cold start. A movie that
  did not occur in training has `is_seen_in_train = false` and zero behavioral
  counts, not invented averages.
- `is_future_release_at_cutoff` must be used as an availability guard in a
  historical experiment. Full MovieLens metadata can describe movies released
  after the training date, but those movies were not recommendable then. The
  source supplies a parsed release year rather than an exact release date, so
  the flag catches years after 2016 but cannot resolve availability within
  2016.
- Genre shares count genre events. One interaction with an Action/Drama movie
  contributes one event to Action and one to Drama.

## Output tables

All paths are under `data/processed/ml-25m/features/` and use Parquet.

### `user_features`

One row per user observed in training. It contains interaction and distinct-item
counts, rating distribution statistics, positive count/rate, first and last
interaction timestamps, recent activity, activity span, recency, and the
feature cutoff.

### `item_features`

One row per movie in the metadata catalog. Training-observed movies contain
interaction/user counts, rating statistics, positive rate, recency, and global
popularity rank/percentile/bucket. All rows retain title, genres, parsed release
year, missing/future-release flags, age at cutoff, and the feature cutoff.

Popularity rank is ordered by training interaction count, then mean rating,
then movie ID for deterministic ties. Bucket 1 is the most popular bucket.

### `user_genre_features`

One row per training user and observed genre. It contains historical and recent
genre counts, rating/positive statistics, and normalized historical/recent
shares. An empty genre list contributes no genre event.

### `validation_context` and `test_context`

The original future interaction columns plus:

- `is_new_user`: user ID was absent from training;
- `is_new_item`: movie ID was absent from training; and
- `cold_start_type`: `known_user_and_item`, `new_user`, `new_item`, or
  `new_user_and_item`.

These tables describe evaluation segments. They do not feed information back
into training features.

## Leakage checks

The integration tests add extreme future interactions and verify that:

1. future users do not appear in `user_features`;
2. a future-only item keeps zero training interactions;
3. a future rating cannot change a training user average; and
4. all four cold-start states are classified from training membership.

The feature pipeline also writes a compact local audit report to
`artifacts/reports/feature_summary.json`.
