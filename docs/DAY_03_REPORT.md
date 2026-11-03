# Day 3 Report: TensorFlow Recommenders Two-Tower Retrieval

## Outcome

Day 3 produced the first learned retrieval stage. Spark creates a deterministic,
coverage-preserving positive sample; PyArrow streams it into `tf.data`; a
TensorFlow Recommenders model trains separate user and movie towers; and the
pipeline exports both SavedModels plus a FAISS-ready item matrix.

The evaluation is honest: it uses future validation positives, removes movies
already seen in training, compares against popularity on exactly the same 500
users, and reports that the current ID-only model does not yet win.

## Results

### Training

| Measurement | Result |
|---|---:|
| Training cutoff | 22 June 2016 05:34:47 UTC |
| Positive threshold | Rating ≥ 4.0 |
| Sample interactions | 1,141,541 |
| Sample/vocabulary users | 137,622 |
| Sample/vocabulary movies | 24,728 |
| Embedding dimension | 64 |
| Batch size | 512 |
| Epochs | 3 |
| Steps per epoch | 2,230 |
| Optimizer | Adagrad, learning rate 0.1 |
| CPU training time | 121.28 seconds |
| Epoch loss | 1637.78 → 1581.47 → 1561.96 |

TFRS reports a summed retrieval loss for the final partial batch, so the
downward trend is useful but its absolute value should not be interpreted as a
mean per-example cross-entropy.

### Matched validation evaluation

| Metric | Two tower | Popularity | Difference |
|---|---:|---:|---:|
| Recall@10 | 0.009723 | 0.020968 | -0.011246 |
| NDCG@10 | 0.017824 | 0.062141 | -0.044316 |
| Recall@50 | 0.042940 | 0.077163 | -0.034223 |
| NDCG@50 | 0.028643 | 0.067186 | -0.038543 |
| Recall@100 | 0.077016 | 0.124161 | -0.047145 |
| NDCG@100 | 0.039770 | 0.081907 | -0.042138 |
| Recall@200 | 0.134912 | 0.187009 | -0.052098 |
| NDCG@200 | 0.059031 | 0.102767 | -0.043736 |

The evaluation contains 500 seeded known users and 13,407 future relevant
items. Exact scoring over 24,728 candidates took 1.07 seconds in batched matrix
operations, or about 469 users per second. This is an offline throughput
measurement, not request latency; Day 4 will benchmark FAISS per-query latency.

### Interpretation

Recall@200 of 0.1349 means the model retrieves about 13.5% of each user's
unseen future-positive movies in its first 200 candidates, on average.

The matched popularity baseline reaches 18.7%. Likely reasons include:

- the current towers use IDs only and cannot exploit genre or activity features;
- only about one hash bucket in ten plus coverage anchors is trained;
- three CPU epochs are a development experiment, not a tuning study; and
- MovieLens has a strong popularity signal that is difficult to beat without
  richer personalization or a hybrid score.

The result is not a failed pipeline. It establishes a reproducible learned
baseline and identifies the next experiments. Shipping a weaker neural model
while hiding the simple baseline would be the actual failure.

### Exported artifacts

The valid run ID is
`cutoff-1466573687-seed-42-20261103T200303Z`. Its generated directory occupies
about 50 MiB and contains:

- user tower SavedModel;
- item tower SavedModel;
- deterministic user and movie vocabularies;
- `(24,728, 64)` contiguous `float32` item embeddings;
- candidate movie IDs aligned row-for-row with the embeddings; and
- a complete JSON training report.

Both SavedModels were loaded after export. Known and OOV IDs returned normalized
`(batch, 64)` vectors, and item norms ranged from approximately 0.9999998 to
1.0000001.

## Concepts learned

### Two towers make large-catalog retrieval possible

The user tower and item tower can run independently. Item vectors are computed
offline once. At request time, the system computes one user vector and searches
for nearby item vectors. A ranker could not afford to run an expensive neural
network over every item in a million-item catalog.

### Embeddings learn coordinates, not human-readable columns

An embedding table assigns each known ID a trainable vector. Training moves
observed user/movie pairs closer and in-batch negatives farther apart. Individual
coordinates usually have no stable human meaning; similarity in the full space
is what matters.

L2 normalization makes the dot product equal cosine similarity and prepares the
matrix for FAISS inner-product search.

### In-batch negatives are efficient but approximate

For a batch of 512 positive pairs, every other movie in the batch becomes a
negative candidate. This avoids explicitly materializing hundreds of negatives
per event, but an observed movie can appear more than once. TFRS accidental-hit
removal prevents the same movie ID from being treated as both the positive and
a negative.

### Sampling must preserve entity coverage

The first naive one-million-row stream covered only 29,049 users because early
Parquet batches were not representative. A model can have a vocabulary entry
without ever updating its embedding, which looks valid but behaves randomly.

The corrected Spark sample keeps one anchor per positive user and item before
adding a deterministic hash sample. It covers all 137,622 learned users and all
24,728 learned candidates.

### Evaluation must match the recommendation policy

The model is evaluated only on future positive interactions. Movies seen in
training are removed from both targets and recommendations. Popularity receives
the same candidates, users, and filter. Without these controls, a comparison can
favor one method for reasons unrelated to model quality.

## Key code walkthrough

### 1. Coverage-preserving Spark sample

[`prepare.py`](../src/recommender/retrieval/prepare.py) filters positive training
events, uses partitioned `row_number` windows to choose user/item anchors, and
uses `xxhash64` for the repeatable 1-in-10 remainder sample.

Unlike a random local slice, this logic is distributed, reproducible, and
guarantees that every vocabulary embedding receives at least one update.

### 2. Bounded Parquet streaming

[`data.py`](../src/recommender/retrieval/data.py) uses PyArrow record batches.
It interleaves Spark part files, converts only user/movie arrays, and yields
bounded NumPy batches to a generator-backed `tf.data.Dataset`.

The TensorFlow pipeline applies `unbatch → shuffle → batch → repeat → prefetch`.
`repeat` is essential when `steps_per_epoch` is explicit; otherwise the input
ends after the first epoch.

The same file builds train-only vocabularies and evaluation users. Validation
targets are filtered to learned IDs, and a full training scan removes already
seen movies.

### 3. User and item towers

[`model.py`](../src/recommender/retrieval/model.py) defines `IdEmbeddingTower`.
`IntegerLookup` maps IDs to dense indices, reserving one OOV index. `Embedding`
stores trainable vectors, and `l2_normalize` produces unit-length outputs.

`TwoTowerRetrievalModel` subclasses `tfrs.models.Model`. Its `compute_loss`
passes user/item vectors and candidate IDs to `tfrs.tasks.Retrieval`, disables
expensive full-catalog metrics during every training batch, and enables
accidental-hit removal.

### 4. Exact offline retrieval

[`evaluate.py`](../src/recommender/retrieval/evaluate.py) computes all item
vectors once, batches users, and uses matrix multiplication for exact scores.
It masks seen-item indices with negative infinity before top-K selection.

The same functions calculate Recall and NDCG for learned and popularity
recommendations. Day 4 will use exact results as the correctness reference for
approximate FAISS indexes.

### 5. Versioned training and export

[`train.py`](../src/recommender/retrieval/train.py) profiles the sample, trains
the model, creates a cutoff/seed/timestamp run ID, exports stable SavedModel
signatures, writes `.npy` arrays, evaluates, and records framework versions,
hashes, hyperparameters, loss, metrics, routing, and artifact paths.

The `latest.json` pointer changes only after a completed run is written.
[`RETRIEVAL_CONTRACT.md`](RETRIEVAL_CONTRACT.md) defines the serving and array
interfaces that FAISS and Triton will consume.

### 6. Explicit cold-start routing

[`fallbacks.py`](../src/recommender/retrieval/fallbacks.py) maps all four
known/new user/item states to two-tower, popularity, content, or combined
strategies. The content implementations are later work, but the boundary is
already explicit and exhaustively tested.

## Complete debugging log

| # | Symptom | Root cause | Fix | Prevention or lesson |
|---:|---|---|---|---|
| 1 | Importing TFRS raised “requires tf.keras to be Keras version 2” | TensorFlow 2.21 defaults to Keras 3 while TFRS 0.7.7 uses `tf-keras` compatibility | Installed the pinned TFRS/`tf-keras` pair and exported `TF_USE_LEGACY_KERAS=1` before TensorFlow imports | Import smoke test and Makefile runtime variable make the compatibility choice explicit |
| 2 | A direct focused Spark test failed with Java `getSubject is not supported` | The direct pytest command bypassed the Makefile and used host Java 25 | Ran project integration commands through `make`, which exports local Java 17 | Use repository commands for Spark; this was an invocation error, not a Spark sampling defect |
| 3 | The first one-million-event profile covered only 29,049 users | Stopping after early Parquet batches biased the sample toward a subset of IDs | Replaced row truncation as the sampling strategy with Spark user/item anchors plus stable hash sampling | Coverage counts are now generated and must equal vocabulary counts |
| 4 | Twelve positive-history movies had metadata years after the cutoff | Parsed title year conflicts with actual pre-cutoff rating timestamps | Treated observed timestamps as stronger availability evidence and retained the movies | The conflict is documented; candidate inclusion requires a positive training event |
| 5 | `train_on_batch` passed a one-element tuple to `compute_loss` | Keras data adaptation wraps a direct dictionary differently from the normal `tf.data` path | Normalized tuple/dictionary inputs in `compute_loss` | Real TFRS integration test exercises `train_on_batch` |
| 6 | The first loss assertion raised an ambiguous NumPy truth-value error | Keras returned multiple logged loss values rather than one scalar | Asserted that all returned values are finite | Tests now accept the documented multi-value training result |
| 7 | Epoch 2 warned that the input ran out of data and performed no training | Explicit `steps_per_epoch × epochs` consumed a finite generator-backed dataset once | Added `.repeat()` after batching and reran the full experiment | Regression test takes two complete cycles and verifies batch sizes `[2, 1, 2, 1]` |
| 8 | Early SavedModel smoke export warned about non-serializable constructor arguments | The subclassed tower did not implement Keras configuration serialization | Stored constructor parameters and implemented `get_config` | Final exports complete without that warning and are loaded in tests |
| 9 | TensorFlow printed CUDA and oneDNN fallback messages | The host has no usable NVIDIA GPU and lacks AVX-512 for an `int64` path | Kept the experiment CPU-only; no correctness change was needed | Reports label the device and distinguish backend warnings from failures |
| 10 | The learned model trails popularity at every K | The first model is ID-only, sampled, and minimally tuned; popularity is strong in MovieLens | Reported the result and retained popularity as the production fallback/reference | Next experiments add richer features, more data, tuning, and hybrid scoring; neural complexity is not assumed to be better |

Two exploratory generated runs remain locally beside the valid run: the 20k
smoke test and the one-epoch run that exposed the missing repeat. They are
ignored by Git, and `latest.json` points only to the valid three-epoch run.

## Tests and quality checks

- 22 automated tests pass in 33.37 seconds on the final local run.
- Ruff and whitespace checks pass.
- Real Spark tests verify positive-only sampling and complete user/item anchor
  coverage.
- PyArrow tests verify positive filtering, deterministic bounds, and sample
  profiling.
- Vocabulary tests reject users/items without positive training evidence and
  exclude validation-only entities.
- Evaluation tests remove training-seen targets and unknown entities.
- A real TFRS batch performs gradient updates; exported SavedModel inference is
  then loaded and checked.
- All four cold-start states are tested.
- Post-training validation checks item array shape, `float32` dtype, vector
  norms, SavedModel signature names, output dimensions, and OOV inference.

Not yet tested: GPU training, multi-host TensorFlow, FAISS indexing, or Triton
inference. These are outside the Day 3 boundary.

## Learning resources, in order

1. [TensorFlow Recommenders basic retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
   — 45 minutes. Focus on query/candidate towers, lookup vocabularies,
   embeddings, the retrieval task, top-K evaluation, and export. Compare its
   small MovieLens example with our temporal and seen-filtered evaluation.
2. [TFRS `Retrieval` task API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval)
   — 15 minutes. Read `temperature`, `num_hard_negatives`,
   `remove_accidental_hits`, `candidate_ids`, and `compute_metrics`.
3. [TensorFlow `tf.data` performance guide](https://www.tensorflow.org/guide/data_performance)
   — 30 minutes. Focus on shuffle-buffer memory, batching, prefetching, and why
   transformation order matters.
4. [TensorFlow SavedModel guide](https://www.tensorflow.org/guide/saved_model)
   — 25 minutes. Focus on signatures, named inputs/outputs, variables, and
   loading without the original Python class.
5. [TFRS efficient serving tutorial](https://www.tensorflow.org/recommenders/examples/efficient_serving)
   — optional 30-minute preview. Focus on the separation between a query tower
   and an ANN candidate index; we will implement the FAISS version next.

## Practice exercises

1. If a batch contains 512 unique positive pairs, how many candidate scores are
   in its in-batch score matrix? Why does doubling batch size cost more than
   twice as much?
2. Vectors `u = [0.6, 0.8]`, `a = [1, 0]`, and `b = [0.6, 0.8]` are normalized.
   Calculate both dot products and identify the nearer item.
3. Explain why a vocabulary ID that never appears in the sample still produces
   a vector but does not produce a learned recommendation.
4. Add one synthetic user with two positive interactions and verify that a
   sampling modulus of 1,000 still retains at least one user anchor.
5. Remove `.repeat()` temporarily and predict the exact warning before running
   the small regression test.
6. Explain why the Day 3 popularity result cannot be compared directly with the
   Day 1 published test Recall@10.
7. Propose one tower feature and explain which cold-start or quality limitation
   it addresses.

### Exercise answer check

- Exercise 1: `512 × 512 = 262,144` scores. Doubling to 1,024 creates four
  times as many pairwise scores.
- Exercise 2: `u·a = 0.6`; `u·b = 1.0`, so `b` is nearer.
- Exercise 3: its random initialization is never updated by a gradient.
- Exercise 4: the `_user_anchor_rank == 1` branch is independent of the hash.
- Exercise 5: Keras says the input ran out of data before all epoch steps.
- Exercise 6: they use different splits and eligible-user populations; only the
  matched validation baseline is a fair Day 3 comparison.
- Exercise 7: genre features help distinguish tastes and can represent a new
  item even before it has collaborative interactions.

## Reproduction commands

From the repository root after Day 2:

```bash
source .venv/bin/activate
make install-retrieval
make prepare-retrieval
make train-retrieval
make test
make lint
```

For a fast end-to-end smoke run:

```bash
python -m recommender.retrieval.train \
  --max-train-interactions 20000 \
  --max-validation-users 20 \
  --epochs 1
```

Generated metrics are in `artifacts/reports/retrieval_metrics.json`; the valid
model location is in `artifacts/models/retrieval/latest.json`.

## Next day

Day 4 will build an exact FAISS inner-product index from the exported normalized
item matrix, verify that its top-K results match NumPy exact search, and compare
it with an approximate HNSW or IVF index.

The quality gate is: index persistence/reload, ID mapping correctness, exact
top-K agreement, approximate recall relative to exact search, index size, build
time, and P50/P95 single-query and batch latency.
