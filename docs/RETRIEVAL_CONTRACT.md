# Retrieval Model Contract

This document defines the Day 3 TensorFlow Recommenders development model,
training inputs, serving outputs, and evaluation rules.

## Training boundary

- The feature and vocabulary cutoff is 22 June 2016 05:34:47 UTC.
- Only training ratings greater than or equal to 4.0 are positive pairs.
- Validation and test interactions never enter the sample or vocabularies.
- User vocabulary contains training users with at least one positive event.
- Candidate vocabulary contains training movies with at least one positive
  event.
- Every lookup has one out-of-vocabulary bucket, but unknown IDs must be routed
  to a fallback rather than treated as personalized recommendations.

Twelve movies have positive events before the cutoff even though their parsed
title years are after 2016. The interaction timestamp is stronger evidence that
the movie was available, so these movies remain in the retrieval vocabulary.
The inconsistency is retained as a documented data-quality case.

## Coverage-preserving sample

`make prepare-retrieval` writes
`data/processed/ml-25m/retrieval/train_sample/`. Spark keeps:

1. the latest positive interaction for every positive-history user;
2. the latest positive interaction for every positive-history movie; and
3. interactions whose stable hash falls in one of 1 out of 10 buckets.

The rules are combined with OR, so an interaction is written once. This gives a
bounded development dataset while guaranteeing representation for every ID that
receives a learned embedding.

Current sample:

| Field | Value |
|---|---:|
| Positive interactions | 1,141,541 |
| Users | 137,622 |
| Movies | 24,728 |
| Minimum rating | 4.0 |
| Latest timestamp | Training cutoff |

## Input pipeline

PyArrow reads Spark's Parquet part files in bounded record batches. Files are
visited in deterministic seeded order and batches are interleaved round-robin.
TensorFlow then unbatches, shuffles through a 100,000-example buffer, batches to
512 pairs, repeats for multiple epochs, and prefetches the next batch.

At no point are all 1.14 million interactions materialized in Python memory.
The small ID vocabularies and exported embedding matrices are intentionally
held in memory.

## Model

The development model has two independent towers:

- user ID → `IntegerLookup` → 64-dimensional embedding → L2 normalization;
- movie ID → `IntegerLookup` → 64-dimensional embedding → L2 normalization.

Their dot product is cosine similarity because both outputs are normalized.
`tfrs.tasks.Retrieval` uses other movies in each batch as negatives and removes
accidental hits when the same positive movie occurs more than once in a batch.
Adagrad trains both towers for three epochs at learning rate 0.1.

This ID-only model is equivalent to a neural matrix-factorization baseline. It
does not yet consume Day 2 activity or genre features; those are planned model
improvements, not silently claimed capabilities.

## Export contract

Each run is written beneath `artifacts/models/retrieval/<run-id>/`. The latest
successful run is identified by `artifacts/models/retrieval/latest.json`.

### User tower

- Format: TensorFlow SavedModel
- Signature: `serving_default`
- Input: `user_id`, `int64`, shape `[batch]`
- Output: `embedding`, `float32`, shape `[batch, 64]`

### Item tower

- Format: TensorFlow SavedModel
- Signature: `serving_default`
- Input: `movie_id`, `int64`, shape `[batch]`
- Output: `embedding`, `float32`, shape `[batch, 64]`

### FAISS inputs

- `candidate_movie_ids.npy`: `int64`, shape `(24,728,)`
- `item_embeddings.npy`: contiguous `float32`, shape `(24,728, 64)`
- Each embedding has L2 norm approximately 1.0.

The row at index `i` in the embedding array belongs to the movie ID at index
`i` in the ID array.

## Offline evaluation

The evaluator selects 500 seeded validation users who:

- are in the learned user vocabulary;
- have positive validation movies in the candidate vocabulary; and
- retain at least one validation target after training-seen movies are removed.

Exact matrix multiplication scores every candidate. Training-seen movies are
assigned negative infinity before top-K selection. Metrics are macro-averaged
over users. The popularity baseline uses the identical users, targets,
candidates, and seen-item filter.

These validation results must not be compared directly with Day 1's published
test metrics because the split segment and eligible-user population differ.

## Cold-start routing

| Evaluation state | Strategy |
|---|---|
| Known user, known item | Two tower |
| New user, known item | Popularity |
| Known user, new item | Content-based |
| New user, new item | Content + popularity |

Content fallbacks are part of a later milestone. The current code makes the
route explicit and tested so unknown IDs are not accidentally presented as
personalized tower results.
