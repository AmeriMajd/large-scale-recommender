# Day 5 Report: Deep Candidate Ranking

## Outcome

Day 5 completes the second modeling stage. Spark constructs deterministic
positive/negative ranking pairs and joins training-cutoff features. Exact FAISS
creates 200-candidate future-validation lists. A Wide & Deep model and a Deep &
Cross Network train on identical streamed examples, stop on chronological
validation AUC, and are compared with retrieval and popularity on identical
candidate lists.

DCN is the selected model. It improves NDCG@10 from the raw retrieval order's
0.0178 to 0.0753 and beats the candidate-set popularity baseline's 0.0713. Both
models are exported with a stable twelve-input SavedModel signature.

## Results

### Example construction

| Measurement | Result |
|---|---:|
| Training positives | 200,000 |
| Training negatives | 594,806 |
| Total training rows | 794,806 |
| Validation users | 500 |
| Candidates per user | 200 |
| Validation candidate rows | 100,000 |
| Positive candidate rows | 1,461 |
| All future relevant items | 13,407 |

Three negatives are requested per positive. The total is slightly below
600,000 because sampled known-positive pairs are rejected and duplicate hash
outcomes are removed. Every negative was checked against the full training
positive-pair table, not only the 200k positive sample.

Only 1,461 positive rows occur inside the 100,000 candidate rows. This exposes
the ranking ceiling: reordering cannot recover future positives absent from the
retrieval stage.

### Matched future-validation ranking

| Order | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| Raw two-tower retrieval | 0.00972 | 0.01782 | 0.04294 | 0.02864 |
| Candidate popularity | 0.02886 | 0.07130 | 0.08374 | 0.06967 |
| Wide & Deep | 0.03029 | 0.07403 | 0.08841 | 0.07438 |
| DCN | **0.03244** | **0.07529** | **0.08885** | **0.07477** |

Relative to raw retrieval, DCN raises NDCG@10 by about 322% and Recall@10 by
about 234%. Relative to popularity, the gains are smaller but meaningful in
this development experiment: about 5.6% in NDCG@10 and 12.4% in Recall@10.
These are offline MovieLens results, not claims about online click-through.

### Training behavior

| Model | Parameters | Restored epoch | Validation AUC | CPU training |
|---|---:|---:|---:|---:|
| Wide & Deep | 2,611,500 | 1 | 0.83088 | 39.46 s |
| DCN | 2,616,959 | 1 | 0.83812 | 40.36 s |

The first uninterrupted run exported the last epoch and produced weaker list
metrics. The final pipeline uses logit-aware AUC, early stopping with patience
one, and best-weight restoration. Both models stopped after epoch two and
restored epoch one.

The valid run ID is
`cutoff-1466573687-seed-42-20261103T200303Z-ranking-seed-42-20261104T223813Z`.
Together, the two SavedModels occupy about 66 MiB. Most parameters are the user
and movie ID embedding tables.

## Concepts learned

### Retrieval and ranking optimize different jobs

Retrieval cheaply reduces 24,728 movies to 200. Ranking can then afford richer
feature interactions for each user/candidate pair. The ranker improves ordering
inside that set, but its maximum Recall is bounded by retrieval coverage.

### Unobserved feedback is not a true negative

MovieLens says what users rated, not every movie they saw and rejected. Random
unobserved movies are useful development negatives, but their label is a
modeling assumption. The anti-join prevents known likes from being mislabeled;
it cannot prove every remaining pair is disliked.

### Hard negatives determine generalization

Training negatives come from the candidate catalog, while validation negatives
are the 200 movies the retrieval model considered most similar. Those are much
harder. The resulting train/validation gap explains why training AUC near 0.98
does not translate to validation AUC near 0.98. A future improvement should
mine training negatives from FAISS retrieval output.

### DCN learns explicit feature interactions

An MLP can learn nonlinear patterns implicitly. DCN adds cross layers that
explicitly multiply the original feature vector with the current crossed
representation. This is useful for patterns such as “high retrieval similarity
matters more for active users” or “popularity interacts with user history.”

### AUC and NDCG answer different questions

AUC checks whether a random positive candidate scores above a random negative.
NDCG emphasizes correct items near the top of each user's list. Early stopping
uses AUC because it is available during every validation pass; final model
selection uses NDCG@10 because the product displays a short ordered list.

### Chronological validation reveals overfitting

Both models improved on training while future-validation AUC declined after the
first epoch. Restoring the best validation epoch preserved the stronger model
state. The time split materially affected the shipped artifact, not just the
report.

## Key code walkthrough

### 1. Deterministic positive and negative pairs

[`prepare.py`](../src/recommender/ranking/prepare.py) deduplicates positive
training pairs, orders them with a stable seeded hash, and retains 200,000.
Each positive is crossed with three negative slots. Another seeded hash maps
each slot into the positive-history item catalog.

The resulting candidates are left-anti-joined with all positive pairs from the
full training split. Duplicate user/movie negatives are removed before labels
and features are written.

### 2. Real future candidate lists

The same preparation module loads the exported user tower and persistent exact
FAISS index. It retrieves enough movies to remove training-seen items and keep
200 candidates per user. Future validation interactions provide labels and
`total_relevant_count`, never model features.

Spark joins user activity, rating behavior, recency, item popularity, item
quality, and age from the Day 2 snapshot. Log/count features are scaled before
storage so serving applies the same transformations.

### 3. Streaming data and retrieval affinity

[`data.py`](../src/recommender/ranking/data.py) streams Parquet record batches
through a generator-backed `tf.data` pipeline. It loads bounded user/item tower
embedding tables once and computes a vectorized dot product for training pairs.
Validation uses the score recorded during actual candidate retrieval.

Inputs are shuffled, batched to 1,024, repeated for training, and prefetched.
The full 794k-row table is never materialized as Python objects.

### 4. Comparable ranking architectures

[`model.py`](../src/recommender/ranking/model.py) uses one `FeatureEncoder` for
the twelve-input contract. It learns 16-dimensional user/movie embeddings and
concatenates them with ten dense features.

`WideAndDeepRanker` sums a linear and MLP logit. `DcnRanker` sends the encoded
vector through two TFRS cross layers and the MLP in parallel, concatenates both
representations, and predicts one relevance logit.

### 5. Honest list metrics and selection

[`evaluate.py`](../src/recommender/ranking/evaluate.py) groups candidates by
user, sorts them by each model or baseline, and macro-averages Recall/NDCG.
Crucially, the denominator includes future relevant items that FAISS missed.

[`train.py`](../src/recommender/ranking/train.py) trains both models with the
same datasets and seed, restores the best validation-AUC epoch, calculates
candidate and list metrics, selects by NDCG@10, exports both models, and updates
`latest.json` only after success.

## Complete debugging log

| # | Symptom | Root cause | Fix | Prevention or lesson |
|---:|---|---|---|---|
| 1 | The first full run's training AUC rose while validation AUC and list quality fell | Easy random training negatives and millions of ID parameters let both models overfit historical pairs | Added chronological validation-AUC early stopping with best-weight restoration and reran both models | Never export the final epoch automatically; record full histories and validate out of time |
| 2 | Restoring best weights changed the comparison between Wide & Deep and DCN | Last-epoch scores compared differently overfit model states | Final selection now occurs after each model's restored checkpoint and uses NDCG@10 | Compare valid checkpoints, not equal epoch numbers |
| 3 | Validation loss was much higher than training loss | Random catalog negatives are easier than retrieval-generated validation negatives, and validation positive prevalence is only 1.46% | Kept the honest validation distribution and reported the gap | Mine FAISS hard negatives in the next model iteration; training metrics are not deployment estimates |
| 4 | Validation binary accuracy was only 0.26–0.34 despite useful ranking metrics | A fixed zero-logit threshold is sensitive to class prevalence and calibration; ranking cares about relative order | Selected with AUC/NDCG and labeled binary accuracy diagnostic-only | Match metrics to the serving decision; calibrate separately if probabilities are needed |
| 5 | Spark warned that candidate-index `row_number` used an unpartitioned window | Assigning a deterministic contiguous index requires a global order | Accepted the single partition for only 24,728 candidate IDs; large interaction joins remain distributed | Replace with a scalable precomputed ID map before using a million-item catalog |
| 6 | TensorFlow logged CUDA initialization failures | The host has no usable NVIDIA GPU | Ran and reported CPU training; outputs remained finite and reproducible | Backend discovery warnings are not model failures, but device must be recorded |
| 7 | SavedModel export warned that subclass constructors contain non-serializable Python objects | Keras configuration serialization differs from TensorFlow SavedModel graph/signature export | Exported explicit signatures and reloaded them without the Python classes | Triton consumes SavedModel signatures; keep reload tests and do not claim `.keras` reconstruction support |
| 8 | Snapshot training metrics can be optimistic | Training labels and aggregate features both come from the broad training window, although validation is strictly later | Documented the limitation and used only future validation for selection | Stronger follow-up: create an inner chronological feature cutoff and label window within `ratings_train` |
| 9 | Final review found AUC configured for probabilities while models emit logits | `keras.metrics.AUC` defaults to probability inputs, but the loss and serving contract intentionally use raw logits | Set `from_logits=True` and reran checkpoint selection and all reported metrics | Losses and metrics must agree on the model-output domain; the integration test now exercises the compiled configuration |

The first generated ranking run remains locally as an ignored artifact. The
`latest.json` pointer names only the rerun with restored best weights.

## Tests and quality checks

- 36 repository tests pass in 36.36 seconds, including real Spark, TensorFlow,
  PyArrow, FAISS, Wide & Deep, DCN, and SavedModel integration tests.
- Spark tests verify sampled negatives never overlap known positive pairs.
- Both model architectures perform a real gradient update with finite results.
- The DCN SavedModel reloads and scores all twelve named feature inputs.
- Metric tests prove retrieval misses remain in Recall's denominator.
- Configuration tests reject evaluation K larger than the candidate list.
- The valid full SavedModel was independently loaded; its signature exposes
  twelve inputs and one `(batch,)` float logit.
- Ruff and whitespace validation run before commit.

Not yet tested: hard-negative mining, a strictly earlier inner feature cutoff,
probability calibration, multi-GPU training, production concurrency, Triton, or
online click-through.

## Learning resources, in order

1. [TFRS basic ranking tutorial](https://www.tensorflow.org/recommenders/examples/basic_ranking)
   — 30 minutes. Focus on ranking inputs, labels, `tfrs.tasks.Ranking`, losses,
   and why ranking follows retrieval.
2. [TFRS Deep & Cross Network tutorial](https://www.tensorflow.org/recommenders/examples/dcn)
   — 40 minutes. Focus on explicit feature crosses, stacked versus parallel
   layouts, low-rank crosses, and interpreting cross-layer weights.
3. [TFRS ranking task API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Ranking)
   — 15 minutes. Compare loss, prediction metrics, label metrics, and sample
   weights with our direct Keras binary-ranking loop.
4. [Google ML recommendation-system overview](https://developers.google.com/machine-learning/recommendation/overview/types)
   — 25 minutes. Connect candidate generation, scoring, and re-ranking to the
   complete architecture.
5. [DCN V2 paper](https://arxiv.org/abs/2008.13535)
   — optional 45 minutes. Read the cross-layer equations, low-rank mixture of
   experts, and production lessons.

## Practice exercises

1. Why are validation candidates harder negatives than random catalog movies?
2. For a user with 20 relevant future movies and three hits in the top ten,
   calculate Recall@10.
3. Explain why the ranker cannot raise Recall@200 above retrieval Recall@200
   when both operate on the same 200 candidates.
4. Change `negatives_per_positive` from three to one. Predict the effect on
   training time, prevalence, and score calibration before running it.
5. Draw the DCN parallel path from encoded features to logit. Which path learns
   explicit interactions?
6. Why did validation AUC select checkpoints while NDCG@10 selected the final
   model?
7. Design an inner chronological split that removes the snapshot-training
   limitation documented above.

### Exercise answer check

- Exercise 1: retrieval deliberately chooses movies already similar to the
  user, while random movies are often obviously irrelevant.
- Exercise 2: `3 / 20 = 0.15`.
- Exercise 3: reordering cannot introduce absent items; only their positions
  change.
- Exercise 4: fewer rows and faster training, higher positive prevalence, and
  logits that need different calibration.
- Exercise 5: both cross and deep paths begin at the encoder; the TFRS cross
  layers learn explicit bounded-degree interactions.
- Exercise 6: AUC is inexpensive inside each epoch; NDCG requires grouped
  candidate lists and matches the final presentation objective.
- Exercise 7: build aggregates from an early part of `ratings_train`, labels
  from a later slice, then retain the existing validation split for selection.

## Reproduction commands

From the repository root after Days 1–4:

```bash
source .venv/bin/activate
make install-retrieval
make install-indexing
make prepare-ranking
make train-ranking
make test
make lint
```

Generated preparation and model metrics are written to
`data/processed/ml-25m/ranking/preparation_report.json` and
`artifacts/reports/ranking_metrics.json`. The valid model run is located by
`artifacts/models/ranking/latest.json`.

## Next day

Day 6 will implement cold-start content/popularity fallbacks and a Redis-backed
serving cache around retrieval plus ranking. The quality gate is deterministic
routing for all known/new states, stable cache keys with model versions, expiry
behavior, graceful Redis failure, and an end-to-end recommendation response.
