# Ranking Model Contract

This document defines the Day 5 boundary between FAISS candidate retrieval and
second-stage relevance scoring.

## Training examples

Training examples are written to
`data/processed/ml-25m/ranking/training_examples/`.

- Positive label `1`: a distinct training user/movie pair with rating ≥ 4.0,
  selected deterministically from the coverage-preserving retrieval sample.
- Negative label `0`: a deterministic hash sample from the train-positive item
  catalog.
- Every negative pair is anti-joined against every positive pair in the full
  training split. “Unobserved” is not claimed to mean “disliked.”
- The default development set contains at most 200,000 positives and three
  requested negatives per positive. Hash collisions and positive-pair removal
  can reduce the final negative count.

User and item aggregates come only from the Day 2 training-cutoff snapshot.
Validation/test rows never alter these features. Because the development
training labels and snapshot aggregates share the training window, training
metrics are optimistic: a stricter production experiment should build an
earlier feature cutoff and a later ranker-label window inside `ratings_train`.
The chronological validation metrics remain out-of-time and are the metrics
used for model selection.

## Validation candidates

The 500 seeded known validation users are the same population used in Day 3.
For each user, exact FAISS retrieves candidates, training-seen movies are
removed, and the first 200 survivors are labeled using future validation
positives. No validation rating becomes a feature.

`total_relevant_count` includes relevant movies that retrieval missed. Recall
therefore remains end-to-end: a ranker cannot receive credit for an item it
never had the opportunity to score.

## Model inputs

Every input is a one-dimensional batch. IDs are `int64`; all other fields are
`float32`.

| Input | Meaning |
|---|---|
| `user_id` | Known training user ID |
| `movie_id` | Known candidate movie ID |
| `user_log_interactions` | `log1p` training interaction count, scaled by 15 |
| `user_mean_rating` | Training mean rating divided by 5 |
| `user_positive_rate` | Fraction of training ratings ≥ 4 |
| `user_log_recency` | `log1p` days since last event, scaled by 12 |
| `item_log_interactions` | `log1p` training interaction count, scaled by 15 |
| `item_mean_rating` | Training mean rating divided by 5 |
| `item_positive_rate` | Fraction of training ratings ≥ 4 |
| `item_popularity_score` | One minus training popularity percentile |
| `item_log_age` | `log1p` age at cutoff, scaled by 8 |
| `retrieval_score` | Dot product of normalized Day 3 tower embeddings |

The rankers learn separate 16-dimensional user and movie ID embeddings. An
unknown ID maps to an OOV bucket, but the serving router must send cold-start
traffic to the documented fallback rather than presenting OOV output as
personalization.

## Models

### Wide & Deep

The encoded feature vector enters a linear logit and a `[128, 64]` ReLU MLP in
parallel. Their logits are added. The linear branch captures direct effects;
the deep branch learns nonlinear combinations.

### Deep & Cross Network

The same encoded vector enters two explicit TFRS DCN cross layers and the same
MLP in parallel. Their outputs are concatenated and projected to one logit.

Both models use binary cross-entropy from logits and Adam. Early stopping
monitors chronological validation AUC and restores the best epoch. Final model
selection uses validation NDCG@10 because ordering quality—not thresholded
classification—is the serving objective.

## Export and response

Every successful run writes both SavedModels below
`artifacts/models/ranking/<run-id>/`. The selected model is recorded in
`artifacts/models/ranking/latest.json`.

The `serving_default` signature accepts the twelve named inputs above and
returns:

- `logit`: `float32`, shape `(batch,)`.

Higher logits rank first. The score is not a calibrated click probability.
Apply a sigmoid only when probability semantics have been separately
calibrated and validated.

## Evaluation

- Candidate AUC evaluates pair ordering across the 100,000 retrieved rows.
- Recall@K divides hits by all future relevant items for that user, including
  retrieval misses.
- NDCG@K uses binary relevance and the same full relevant count for its ideal
  list.
- Retrieval order, popularity order, Wide & Deep, and DCN use identical users,
  candidate rows, labels, and features.
- Binary accuracy is diagnostic only. Its fixed threshold is not comparable
  across training and validation because candidate prevalence differs sharply.
