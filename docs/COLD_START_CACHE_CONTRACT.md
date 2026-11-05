# Cold-Start and Cache Contract

This document defines the Day 6 behavior for users who do not have a learned
user embedding and for recommendation-result caching.

## Routing rules

| Request state | Strategy |
|---|---|
| Known training user | Two-tower retrieval followed by the assigned ranker |
| Unknown user, no preferences | Diversified training-era popularity |
| Selected genres | Genre affinity blended with popularity and mean rating |
| Liked movies | Temporary genre profile inferred from those movies |

Unknown users never pass through the learned user-ID tower. Unknown genres and
movie IDs are ignored safely. Liked movies are excluded from the response.

## Scores and diversity

The catalog contains only items seen in the training period with at least one
positive interaction. Popularity is `1 - popularity_percentile` and quality is
`mean_rating / 5`, both fitted at the training cutoff.

- No preferences: `0.85 * popularity + 0.15 * quality`.
- Preferences: `0.65 * genre_affinity + 0.25 * popularity + 0.10 * quality`.
- At most three initial selections use the same deterministic genre bucket;
  deferred candidates fill any remaining positions.
- Ties are broken by ascending movie ID, making repeated requests reproducible.

These are heuristic scores, not click probabilities.

## Cache keys and values

Keys contain the namespace, a SHA-256 prefix of the retrieval/ranker run IDs,
the assigned variant, subject, requested limit, and a canonical context hash.
Changing a model version, ranker assignment, genres, liked movies, or response
size therefore creates a new key. Values are compact JSON and expire after 900
seconds by default.

Redis failures are fail-open: recommendation computation continues, while hit,
miss, read-error, and write-error counters make the degraded state visible.
The health response reports whether Redis answers `PING`. The in-memory backend
exists for deterministic tests and local benchmarks; it is not shared between
processes.

## Privacy and invalidation

Subjects are application-level user/session identifiers. Do not place emails,
names, or other personal data in cache keys. New model run IDs invalidate old
entries by namespace rather than requiring a broad destructive delete. Expired
values may remain physically present until Redis evicts them, but are not read.

