# Recommendation API and Experiment Contract

## Start the local service

```bash
source .venv/bin/activate
make install-retrieval install-indexing install-serving
make serve
```

The default address is `http://127.0.0.1:8000`; interactive OpenAPI docs are at
`/docs`, and the Recommender Lab interface is served at `/`. Startup loads the
latest successful retrieval, exact FAISS, and both ranking SavedModel artifacts.
The HTTP contract is independent of whether local TensorFlow or Triton provides
inference.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Artifact identity, catalog sizes, and Redis health |
| `GET /recommendations/{user_id}?limit=10` | Known-user two-stage recommendations; unknown IDs route to cold start |
| `POST /recommendations/cold-start` | Session recommendations from genres and/or liked movie IDs |
| `GET /catalog/search?q=toy&limit=8` | Bounded title search for the cold-start movie picker |
| `POST /events/impression` | Explicitly record an externally displayed list |
| `POST /events/click` | Record a click only when its movie appeared in that impression |
| `GET /experiments/{experiment_id}/metrics` | Per-variant impressions, unique clicked impressions, CTR, and position-discounted clicks |

Recommendation responses include `subject`, `variant`, `strategy`,
`model_version`, `cache_hit`, `impression_id`, and an ordered recommendation
array. Ranker scores are logits, not calibrated probabilities. The engine also
records an impression automatically for each generated response.

## Cold-start request example

```json
{
  "session_id": "demo-123",
  "selected_genres": ["Action", "Comedy"],
  "liked_movie_ids": [1],
  "limit": 10
}
```

## A/B semantics

`SHA-256(experiment_id + subject_id)` deterministically assigns each known
subject to `wide_and_deep` or `dcn`. Assignment is sticky while the experiment
ID is unchanged and independent of process state. The model variant is part of
the cache key, preventing cross-variant contamination.

Cold-start impressions use the separate `cold-start-fallback` experiment ID
and cannot enter ranker metrics. CTR counts at most one clicked impression per
served impression, even if duplicate click events arrive. The position metric
weights that first click by `1/log2(rank+1)`.

When Redis is healthy, events are appended to a bounded Redis Stream and each
impression is stored for seven days to validate clicks. When it is unavailable,
the single-process demo uses an in-memory event store. The metrics endpoint
scans the bounded stream and is suitable for the demo, not high-volume
analytics; production aggregation belongs in a stream processor or warehouse.

MovieLens contains ratings rather than impression/click logs. Any events sent
to this API are demo or real UI events and must not be described as historical
MovieLens CTR.

## Error behavior

- Invalid request sizes or missing required fields return `422`.
- Unknown/expired impression IDs return `404` on click.
- A clicked movie absent from its impression returns `422`.
- Redis cache errors do not fail recommendation requests.
- Artifact-loading errors fail startup instead of silently serving a partial
  model stack.
