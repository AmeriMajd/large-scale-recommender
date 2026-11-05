# Day 7 Report: End-to-End API and A/B Experiment

## Outcome

Day 7 connects all completed stages into one service: load a user tower, query
the persisted exact FAISS index, filter training-seen movies, assemble the same
twelve ranker inputs used on Day 5, score with the subject's assigned Wide &
Deep or DCN model, cache the ordered list, and return a versioned response.

FastAPI exposes health, known-user, cold-start, impression, click, and
experiment-metrics routes. Deterministic hashing makes A/B assignment sticky.
Redis Streams store validated events when Redis is available; the local demo
falls back to an in-memory store without claiming persistence.

## Local benchmark

The benchmark used the real exported models and exact index through FastAPI's
in-process client. It used user 1, ten results, 30 warm repetitions, one CPU,
and an in-memory cache after detecting that local Redis was unavailable.

| Path | First miss | Warm P50 | Warm P95 |
|---|---:|---:|---:|
| Known user: tower + FAISS + DCN | 374.10 ms | 13.79 ms | 15.62 ms |
| Genre cold start | 119.94 ms | 14.41 ms | 16.41 ms |

The cache recorded 60 hits and two misses with no read/write errors. These are
development-machine latency measurements, not concurrent production load or a
Triton benchmark. The machine-readable result is
`artifacts/reports/serving_benchmark.json` (ignored from Git).

## Concepts learned

Training-serving skew occurs when online code transforms features differently
from training. `FeatureStore.ranker_inputs` deliberately repeats Day 5's field
names, scaling constants, dtypes, and retrieval-score semantics.

A/B assignment happens before cache lookup, and the variant is in the cache
key. Otherwise one model's cached list could be returned to the other group.
Changing the experiment ID reshuffles assignment intentionally.

An impression is the denominator of CTR. A click is accepted only for a movie
that was actually served, and duplicate click events count once in CTR. This
prevents impossible clicks and retry traffic from inflating the headline
metric.

## Key code walkthrough

[`engine.py`](../src/recommender/serving/engine.py) is the orchestration layer.
It loads artifact pointers, performs retrieval/filtering/ranking, routes unknown
users, reads/writes the cache, and records impressions. The API does not contain
model logic.

[`experiment.py`](../src/recommender/serving/experiment.py) implements stable
hash assignment, impression/click validation, unique-impression CTR, the local
store, and the Redis Stream adapter.

[`api.py`](../src/recommender/serving/api.py) defines typed request validation
and HTTP status mapping. Its engine protocol enables fast API tests without
loading TensorFlow.

[`benchmark.py`](../src/recommender/serving/benchmark.py) loads the real engine,
uses an in-memory cache to make misses/hits repeatable, calls the HTTP app, and
writes P50/P95 evidence.

## Complete debugging log

| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| The newest FastAPI/Starlette test stack warned that `httpx` was deprecated and expected `httpx2` | A just-released dependency line changed its test-client integration | Pinned FastAPI 0.121.0, Starlette's compatible transitive range, and httpx 0.28.1 | Reproducible service tests need compatible version sets, not loose newest-version ranges |
| The API test still appeared to hang after the dependency fix | The restricted runner blocks the test client's in-process portal/loopback behavior | Re-ran the same local-only integration tests with loopback permission; they completed in 0.45 s | Distinguish package incompatibility from sandbox networking |
| Importing the API in tests initially risked loading TensorFlow and all artifacts | Engine construction was coupled to module import | Moved `build_default_engine` import and construction into `main`; tests inject a fake engine | Keep application factories lightweight and dependency-injected |
| Cold-start traffic appeared in the ranker experiment | All automatic impressions originally shared one experiment ID | Route fallback impressions to `cold-start-fallback` | Experiment denominators must contain only eligible traffic |
| Repeated click events could make CTR exceed 100% | Metrics counted raw click events | Deduplicate by impression and count the first click for CTR/discounting | Define event and metric semantics before interpreting dashboards |
| TensorFlow printed CUDA initialization errors | The development host has no usable CUDA driver | Used CPU and recorded the environment; inference completed successfully | Device discovery noise is not a failed request, but deployment hardware must be explicit |
| Redis health was false at benchmark startup | No local Redis server was running | Kept fail-open service behavior and used the deterministic in-memory backend for cache latency evidence | Report the backend actually tested; do not pretend in-memory latency is Redis latency |

## Tests and evidence

- Focused Days 6–7 suite: 19 tests passed in 0.45 seconds.
- Full repository suite: 49 tests passed in 34.43 seconds.
- API tests cover health, known/cold recommendations, request limits, impression
  to click to metrics, unknown impressions, and invalid clicks.
- Assignment is stable and falls between 45/55 over 1,000 synthetic subjects.
- The real engine returned ten recommendations for a known user and the
  benchmark confirmed subsequent cache hits.
- Ruff passes across the repository.

One transitive Starlette/AnyIO deprecation warning remains in the test client;
it does not affect behavior and is recorded rather than hidden.

The service has not yet been load-tested under concurrent traffic, connected
to a real Redis process in CI, protected with authentication/rate limiting, or
served through Triton. Those are not hidden as completed work.

## Learning resources, in order

1. [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
   — focus on path/query/body validation, dependency injection, and OpenAPI.
2. [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
   — compare its `TestClient` pattern with our injected fake engine.
3. [Redis Streams introduction](https://redis.io/docs/latest/develop/data-types/streams/)
   — learn append-only events, IDs, trimming, and consumer groups.
4. [Kohavi et al.: controlled experiments on the web](https://doi.org/10.1007/s10618-008-0114-1)
   — learn assignment units, guardrails, novelty effects, and statistical
   power before drawing conclusions from CTR.

## Practice exercises

1. Trace one known-user request from HTTP path to final ordered movie IDs.
2. Why must ranker assignment occur before cache lookup?
3. Add an idempotency key so a retried click is rejected at write time rather
   than deduplicated only at aggregation time.
4. Replace the exact index with HNSW and benchmark quality plus latency.
5. Design an experiment exposure rule for a user who uses several devices.
6. Explain why this project cannot report a trustworthy treatment effect yet.

## Reproduction

```bash
source .venv/bin/activate
make install-retrieval install-indexing install-serving
make serve
# In another terminal:
curl http://127.0.0.1:8000/health
make benchmark-serving
pytest tests/unit/test_experiment.py tests/integration/test_api.py
```

Request/response and experiment semantics are in
[`API_CONTRACT.md`](API_CONTRACT.md).
