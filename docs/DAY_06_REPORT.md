# Day 6 Report: Cold Start and Redis Caching

## Outcome

Day 6 makes recommendation behavior explicit when personalization is
impossible. Unknown users receive a deterministic popularity fallback; genre
choices or liked movies build a temporary content profile. Every path uses
training-cutoff item features and adds simple genre diversity. Recommendation
results can be cached in Redis with versioned, context-aware keys and safely
recomputed when Redis is unavailable.

The real MovieLens feature snapshot produced a 24,728-item eligible catalog.
The no-preference smoke test placed *The Shawshank Redemption*, *The Godfather*,
and *The Usual Suspects* first. Selecting Action and Comedy changed the first
results to content-matching titles, demonstrating that preferences affect the
route rather than merely changing a response label.

## Concepts learned

Cold start is a routing problem as much as a modeling problem. An ID embedding
has no history for an unseen user, so an OOV vector should not be presented as
personalization. A transparent heuristic based on known item content is safer.

Cache correctness depends on the key. A key containing only `user_id` can
return the wrong list after a model deployment, A/B reassignment, preference
change, or different requested size. This project hashes all output
dependencies and relies on TTL for freshness.

Fail-open means Redis is an optimization, not a single point of failure. A
cache connection error increases latency, but it does not prevent the model
from returning recommendations.

## Key code walkthrough

[`cold_start.py`](../src/recommender/serving/cold_start.py) loads only eligible
training-era items, canonicalizes genre names, builds weights from explicit
genres and liked-movie metadata, blends affinity/popularity/quality, and applies
a deterministic diversity pass.

[`cache.py`](../src/recommender/serving/cache.py) separates the cache policy
from the backend. `RecommendationCache` handles JSON, TTL, counters, and
fail-open errors. `RedisKeyValueBackend` sets short connection/command timeouts;
`InMemoryKeyValueBackend` gives tests a Redis-like clock-controlled TTL.

[`config.py`](../src/recommender/config.py) validates TTL, response/candidate
sizes, Redis URL, stream size, host, port, namespace, and experiment ID before
the service loads expensive model artifacts.

## Complete debugging log

| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| Serving dependency installation initially failed with DNS errors | The restricted build environment blocks package-network access | Re-ran the declared installation with approved network access and pinned versions | Treat environment restrictions separately from code failures |
| A long model run ID could produce an unnecessarily long Redis key | The first key design embedded the full retrieval and ranking IDs | Replaced the run-ID segment with a stable SHA-256 prefix | Cache keys should be complete but bounded |
| Redis was not running during the local benchmark | Redis is optional on the development host | Health reports `false`; reads/writes fail open; the benchmark swaps in the in-memory TTL backend | A cache outage must degrade performance, not availability |
| Genre diversity could leave fewer than the requested number of movies | A strict per-genre cap can exhaust eligible buckets | Deferred over-cap candidates fill remaining positions after the diverse first pass | Diversity constraints need a defined relaxation policy |
| Unknown genres and liked IDs produced no usable profile | External input need not match the catalog | Ignore unknown values and fall back to diversified popularity | Validate safely and always define the empty-profile route |

## Tests and evidence

- Unit tests cover model/variant/context key invalidation, bounded key length,
  TTL expiry, cache counters, and fail-open behavior.
- Cold-start tests cover deterministic popularity, genre and liked-movie
  profiles, exclusion, unknown preferences, and diversity.
- Configuration tests reject inconsistent serving sizes and invalid Redis URLs.
- The real catalog smoke test loaded 24,728 movies and returned five finite,
  deterministic results for each of the three fallback modes.

Redis persistence, eviction under memory pressure, a clustered deployment, and
concurrent stampede protection are intentionally not claimed yet.

## Learning resources, in order

1. [Google: Recommendation systems overview](https://developers.google.com/machine-learning/recommendation/overview)
   — connect candidate generation, scoring, and re-ranking to fallback routing.
2. [Redis data types tutorial](https://redis.io/docs/latest/develop/data-types/)
   — learn keys, strings, expiry, and streams before treating Redis as a black
   box.
3. [redis-py guide](https://redis.io/docs/latest/develop/clients/redis-py/)
   — compare connection setup, command execution, and production client
   options with our small adapter.
4. [Netflix: The Netflix Recommender System](https://dl.acm.org/doi/10.1145/2843948)
   — optional deeper reading on blending multiple recommendation approaches.

## Practice exercises

1. Explain why an OOV user embedding is not equivalent to a popularity list.
2. Add a `release_year` preference without using data after the training
   cutoff. Which cache-key field must change?
3. What happens to latency and correctness when Redis goes down?
4. Design a diversity rule for multi-genre movies that is fairer than using
   the alphabetically first genre.
5. Add a unit test proving a liked movie never appears in its own fallback.

## Reproduction

```bash
source .venv/bin/activate
make install-serving
pytest tests/unit/test_cache.py tests/unit/test_cold_start.py
```

The full behavior and key semantics are in
[`COLD_START_CACHE_CONTRACT.md`](COLD_START_CACHE_CONTRACT.md).

