# Day 4 Report: Persistent FAISS Candidate Retrieval

## Outcome

Day 4 turns the Day 3 item embeddings into two versioned, persistent FAISS
indexes. The pipeline builds an exhaustive inner-product baseline and an HNSW
approximate index, stores real movie IDs inside them, reloads both artifacts,
verifies identical pre/post-save results, and benchmarks them with 500 actual
user-tower queries.

The main lesson is empirical: on a 24,728-item development catalog, high-recall
HNSW is roughly tied with exact search. A lower `efSearch` is much faster but
loses candidates. Approximate search becomes compelling as the catalog grows;
it is not automatically superior at every scale and quality target.

## Results

### Inputs and protocol

| Measurement | Result |
|---|---:|
| Source retrieval run | `cutoff-1466573687-seed-42-20261103T200303Z` |
| Candidate movies | 24,728 |
| Embedding dimension | 64 |
| Item dtype | `float32` |
| Item norm range | 0.9999998–1.0000001 |
| Seeded known-user queries | 500 |
| Requested candidates | 200 |
| CPU threads | 1 |
| Warm-up queries | 25 |
| Timed repetitions | 3 |
| FAISS | 1.15.1, CPU |

Every query comes from a known user ID passed through the exported Day 3 user
tower. Single-query timings include the Python binding and `index.search`, but
exclude user inference, network, filtering, ranking, and serialization.

### Exact and approximate operating points

| Index / `efSearch` | Recall@200 vs exact | P50 ms | P95 ms | Batch queries/s |
|---|---:|---:|---:|---:|
| Exact Flat | 1.0000 | 0.2932 | 0.3196 | 3,622 |
| HNSW 64 | 0.7993 | 0.1306 | 0.1566 | 8,197 |
| HNSW 128 | 0.9402 | 0.1824 | 0.2230 | 5,868 |
| HNSW 256 | 0.9859 | 0.2839 | 0.3307 | 3,680 |
| HNSW 512 | 0.9964 | 0.5963 | 1.1760 | 1,723 |

The automatic rule chooses the smallest measured `efSearch` with at least 95%
overlap, so the persisted HNSW default is 256. It recovers 98.59% of exact
top-200 candidates. Its median is about 3% faster than Flat, while its P95 is
about 3% slower; those small differences are effectively parity for this local
experiment.

At `efSearch=128`, HNSW has a clearer speed/quality compromise: 94.02% overlap,
38% lower median latency, and 62% higher batched throughput than exact search.
Whether that loss is acceptable must be decided with downstream ranking and
business metrics, not ANN recall alone.

### Construction and storage

| Measurement | Exact Flat | HNSW |
|---|---:|---:|
| Build time | 0.011 s | 4.249 s |
| File size | 6.23 MiB | 12.64 MiB |
| Reload verified | Yes | Yes |

HNSW is larger because it stores the original vectors plus graph links. That is
expected, and it reinforces that HNSW trades additional build time and memory
for a tunable search path. Both files have SHA-256 checksums in the generated
report.

## Concepts learned

### Exact search is the reference, not the enemy

`IndexFlatIP` scores every candidate, so it has no approximation loss. It is the
ground truth for HNSW overlap and is a reasonable production option at this
development scale. Keeping it also makes future index changes testable.

### Normalized inner product is cosine similarity

For unit vectors, `u · v = cosine(u, v)`. The two towers normalize their output,
so FAISS can use maximum inner product without computing vector norms during
search. Non-normalized data would change the meaning of the score, so the build
fails fast when norms are outside tolerance.

### HNSW exposes a speed/quality dial

HNSW builds a multi-layer neighbor graph. `M` controls graph connectivity,
`efConstruction` controls work during building, and `efSearch` controls how
widely the graph is explored per query. Higher search depth usually improves
overlap but costs latency, as the measured curve shows.

### ANN recall and recommender recall are different

The Day 4 recall asks, “How many of exact FAISS's top 200 did HNSW recover?” The
Day 3 Recall@200 asks, “How many future positive movies did recommendations
recover?” An ANN index can have near-perfect overlap while the underlying model
is weak. Index quality cannot repair model quality.

### IDs must survive the vector boundary

FAISS naturally works with vector row positions. `IndexIDMap2` associates each
row with its real MovieLens ID, preventing a dangerous class of bugs where row
17 is mistaken for movie ID 17. Arbitrary-ID integration tests and reload tests
protect this contract.

## Key code walkthrough

### 1. Input validation and index construction

[`faiss_index.py`](../src/recommender/indexing/faiss_index.py) validates shapes,
types, uniqueness, finite values, and L2 norms before any index is created.
`build_exact_index` wraps `IndexFlatIP` in `IndexIDMap2` and calls
`add_with_ids`. `build_hnsw_index` does the same for `IndexHNSWFlat` after
setting the construction depth.

This module is deliberately independent of TensorFlow and project paths. Small
arrays can exercise the same production primitives in milliseconds.

### 2. Approximation metric and benchmark

`approximate_recall_at_k` computes set overlap per query, then macro-averages
the result. Order within the top K does not affect this index-completeness
metric.

`benchmark_index` performs a warm-up, times every single-query call for P50/P95
and P99, then times whole query batches for throughput. It repeats each mode
three times. One FAISS OpenMP thread makes operating points easier to compare
and avoids hiding per-request cost behind parallel CPU usage.

### 3. Versioned end-to-end build

[`build.py`](../src/recommender/indexing/build.py) follows the retrieval
`latest.json` pointer, loads the SavedModel user tower, deterministically samples
known users, and builds both indexes. It saves and reloads each index, compares
scores and IDs, sweeps four HNSW search depths, selects the quality-gated point,
and writes checksums plus measurements to `index_report.json`.

The global `artifacts/indexes/latest.json` pointer is written only after every
validation and report step succeeds.

### 4. Configuration and command boundary

[`base.toml`](../configs/base.toml) contains the query count, K, thread count,
HNSW parameters, operating points, and seed. [`config.py`](../src/recommender/config.py)
validates positive values and requires `efSearch` values to be unique and
increasing. `make build-index` is the reproducible entry point.

### 5. Tests that target costly failure modes

[`test_faiss_index.py`](../tests/integration/test_faiss_index.py) verifies exact
results against NumPy matrix multiplication, arbitrary external IDs, disk
round-trips, high HNSW overlap, and rejection of duplicates, zero vectors, and
NaNs. [`test_indexing_metrics.py`](../tests/unit/test_indexing_metrics.py)
checks metric semantics and bad shapes.

## Complete debugging log

| # | Symptom | Root cause | Fix | Prevention or lesson |
|---:|---|---|---|---|
| 1 | Ruff stopped on one 89-character line | The new function signature exceeded the repository's 88-column rule by one character | Wrapped the signature over three lines | `make lint` runs before committing; this was formatting only |
| 2 | TensorFlow printed CUDA initialization errors while making benchmark queries | The installed TensorFlow binary probes CUDA, but this machine has no usable NVIDIA GPU | Kept the run explicitly CPU-only; the SavedModel returned correct normalized vectors | Reports record CPU execution; backend discovery messages are not treated as failed inference |
| 3 | HNSW at the selected 98.59%-overlap point was not materially faster than Flat | The catalog is only 24.7k items, K is large at 200, and high `efSearch` explores much of the graph | Kept both artifacts and reported the full trade-off curve instead of claiming a speedup | Always benchmark at the real catalog size, K, quality target, thread count, and query distribution |
| 4 | HNSW used about twice the disk space of Flat | Full-vector HNSW adds graph-neighbor storage rather than compressing vectors | Recorded both sizes and retained Flat for this small catalog | Choose an index based on scale and constraints; use IVF/PQ later if memory becomes the limiting resource |

No source-data corruption, ID mismatch, failed reload, or dependency conflict was
observed. The pinned `faiss-cpu` 1.15.1 wheel installed successfully in the
project virtual environment.

## Tests and quality checks

- Fast indexing/config suite: 23 tests pass.
- Full repository suite: 31 tests pass in 28.72 seconds, including real Spark,
  TensorFlow, PyArrow, and FAISS integration tests.
- Exact FAISS top-10 IDs and scores match NumPy inner-product top-10 on random
  normalized vectors.
- Arbitrary IDs `101`, `305`, and `999` remain correct after disk reload.
- HNSW reaches at least 95% top-20 overlap in a deterministic synthetic test.
- Both full indexes reproduce the same first ten query scores and IDs after
  serialization.
- Generated index and source files have SHA-256 hashes.
- Ruff checks all source and tests.

Not yet tested: million-item catalogs, concurrent request contention, ARM or
GPU FAISS, network-serving latency, seen-item post-filtering, or live traffic.

## Learning resources, in order

1. [FAISS getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)
   — 20 minutes. Focus on `add`, `search`, matrix shapes, returned distances,
   and IDs. Re-create the tiny exact-index test by hand.
2. [FAISS metric types and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
   — 15 minutes. Focus on why normalized vectors turn inner product into cosine
   similarity and why both database and query vectors must be normalized.
3. [FAISS index-selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
   — 25 minutes. Compare Flat, HNSW, IVF, and PQ by accuracy, memory, training,
   update, and scale constraints.
4. [FAISS index reference](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
   — 25 minutes. Focus on `M`, `efConstruction`, `efSearch`, `IDMap`, and the
   storage formulas.
5. [HNSW paper](https://arxiv.org/abs/1603.09320)
   — optional 45 minutes. Read the abstract, skip-list analogy, graph search
   algorithm, and evaluation plots before diving into proofs.

## Practice exercises

1. Why does `IndexFlatIP` need `IndexIDMap2` in this project?
2. Calculate the cosine similarity of normalized vectors `[0.6, 0.8]` and
   `[0.8, 0.6]`. Would `IndexFlatIP` rank this above a score of 0.9?
3. Change the benchmark to `K=50`. Predict what happens to Flat latency and the
   HNSW quality/speed curve, then run it without replacing the main report.
4. Explain why `efSearch=512` is both more accurate and slower than 128.
5. Add an integration case with duplicate movie IDs and explain why silently
   accepting them would make recommendation debugging difficult.
6. Why can Day 4's 98.59% recall coexist with Day 3's 13.49% Recall@200?
7. Sketch how a service should retrieve candidates when 80 of its first 200 are
   already seen by the user.

### Exercise answer check

- Exercise 1: Flat otherwise returns zero-based row offsets, not movie IDs.
- Exercise 2: the dot product is `0.96`, so yes.
- Exercise 3: measure the answer; expect less selection work and a different
  HNSW `efSearch` requirement, but do not claim a result before running it.
- Exercise 4: it explores more graph candidates before finalizing neighbors.
- Exercise 5: one ID could refer to multiple vectors and create duplicate or
  ambiguous search results.
- Exercise 6: one measures ANN agreement with exact model scores; the other
  measures model recommendations against future user behavior.
- Exercise 7: over-fetch more than 200, remove seen/ineligible IDs, and send the
  first 200 survivors to ranking; use a fallback if too few remain.

## Reproduction commands

From the repository root after a successful Day 3 run:

```bash
source .venv/bin/activate
make install-indexing
make build-index
make test
make lint
```

For a smaller smoke benchmark:

```bash
python -m recommender.indexing.build --query-count 25 --k 20
```

Generated measurements are in `artifacts/reports/faiss_metrics.json`; the
versioned indexes are located by `artifacts/indexes/latest.json`.

## Next day

Day 5 will create ranking examples from temporally valid positive and negative
candidates, then train a deep ranker that re-scores retrieval output. The
quality gate is leakage-safe labels/features, reproducible negatives, ranking
metrics against simple scores, exported inference, and a documented candidate
feature contract.
