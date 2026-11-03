# FAISS Index Contract

This document defines the Day 4 boundary between the TensorFlow retrieval model
and candidate search.

## Source artifacts

`make build-index` reads the successful retrieval run named by
`artifacts/models/retrieval/latest.json`:

- `candidate_movie_ids.npy`: unique `int64`, shape `(N,)`;
- `item_embeddings.npy`: finite, contiguous `float32`, shape `(N, D)`; and
- `user_tower/`: SavedModel whose `serving_default` signature maps `user_id`
  batches to normalized `float32` vectors of shape `(batch, D)`.

Rows in the two NumPy files must align. Item and query vectors must have L2 norm
approximately one. The index builder rejects duplicate IDs, zero vectors,
non-finite values, shape mismatches, and non-normalized embeddings.

## Similarity definition

Both indexes use maximum inner product. Because the towers return unit vectors,
the inner product equals cosine similarity. A larger FAISS score means the item
is more similar to the user query.

## Indexes

| File | FAISS type | Purpose |
|---|---|---|
| `exact_ip.faiss` | `IndexIDMap2(IndexFlatIP)` | Exhaustive correctness baseline |
| `hnsw_ip.faiss` | `IndexIDMap2(IndexHNSWFlat)` | Approximate production experiment |

`IndexIDMap2` stores MovieLens movie IDs inside each index. Search therefore
returns real movie IDs, not embedding row offsets. HNSW uses `M=32`,
`efConstruction=200`, and the selected `efSearch` recorded in the report.

HNSW does not support removing vectors. A changed catalog or model export should
produce a new versioned index rather than mutating the old one in place.

## Persistence and provenance

Each run is written to `artifacts/indexes/<run-id>/`. The latest complete run is
identified by `artifacts/indexes/latest.json`. Its `index_report.json` records:

- source retrieval run and SHA-256 hashes;
- FAISS version, vector count, dimension, and norm range;
- construction parameters, build time, file size, and index checksum;
- benchmark query protocol and exact/HNSW performance; and
- successful persistence/reload verification.

Only load index files produced by this pipeline from a trusted location. Verify
the report checksum before deployment; index deserialization is not a substitute
for artifact authentication.

## Search response

For a query matrix shaped `(B, D)`, `index.search(queries, K)` returns:

- scores: `float32`, shape `(B, K)`, descending inner product; and
- movie IDs: `int64`, shape `(B, K)`.

The benchmark requests 200 raw candidates. Seen-item exclusion, business rules,
and cold-start routing are outside the FAISS index. A serving layer should
over-fetch, remove ineligible items, and then pass the remaining candidates to
the ranking model. Unknown users must use the fallback policy instead of the
user tower's OOV embedding.

## Evaluation semantics

`Recall@200 vs exact` is the mean fraction of exhaustive top-200 movie IDs also
returned by HNSW. It measures approximation loss only. It is not the offline
recommendation Recall@200 from Day 3, which compares recommendations with future
positive interactions.

Latency is measured around an in-process Python `index.search` call on one CPU
thread after warm-up. It excludes network transport, model inference, filtering,
ranking, Redis, and serialization. Batched throughput uses the fastest of three
warm runs and must not be presented as end-to-end request throughput.
