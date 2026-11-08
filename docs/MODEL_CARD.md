# Model card

## Model details

This project is a two-stage movie recommender trained on MovieLens 25M:

1. an ID-based TensorFlow Recommenders two-tower model retrieves candidates;
2. Wide & Deep and Deep & Cross Network rankers score those candidates;
3. FAISS performs nearest-neighbor search, with popularity and content-based
   fallbacks for cold-start traffic.

The project is an educational, production-style reference implementation. It
is not a production recommendation service and has not been validated for
commercial decision-making.

## Intended use

- Demonstrating temporal data splitting, leakage controls, retrieval, ranking,
  approximate nearest-neighbor search, model serving, caching, and offline
  evaluation.
- Reproducing the documented MovieLens experiments.
- Providing a portfolio demo with clearly labelled frozen responses when the
  trained backend is unavailable.

It must not be used for high-impact decisions, safety-critical applications,
or claims about real user preferences without additional validation.

## Training data

The source dataset is MovieLens 25M. Interactions are split globally by time,
and features are fitted using training data only. The feature cutoff is 22 June
2016 UTC. Raw data is downloaded from GroupLens and is not redistributed by
this repository.

See [DATA_CONTRACT.md](DATA_CONTRACT.md),
[FEATURE_CONTRACT.md](FEATURE_CONTRACT.md), and the dataset's own license and
README before reuse.

## Evaluation

The development two-tower model does not beat the matched popularity baseline:

| Metric | Two tower | Popularity |
|---|---:|---:|
| Recall@10 | 0.0097 | 0.0210 |
| Recall@50 | 0.0429 | 0.0772 |
| Recall@100 | 0.0770 | 0.1242 |
| Recall@200 | 0.1349 | 0.1870 |

Within the retrieved candidate set, the DCN ranker achieved Recall@10 of
0.0324 and NDCG@10 of 0.0753, modestly ahead of the Wide & Deep ranker and the
popularity ordering. These are offline development results, not online-lift
measurements.

Detailed protocols and machine-specific latency context are documented in
[RETRIEVAL_CONTRACT.md](RETRIEVAL_CONTRACT.md),
[RANKING_CONTRACT.md](RANKING_CONTRACT.md), and the day reports.

## Limitations

- MovieLens ratings are explicit feedback from a historical, self-selected
  population; they do not represent all viewers or current catalogs.
- The retrieval towers use IDs only, limiting generalization and cold-start
  quality.
- Offline relevance treats high future ratings as positives and does not model
  impressions, position bias, non-click exposure, satisfaction, or business
  constraints.
- The demo's A/B events are simulated or local interactions, not evidence of
  causal online lift.
- Ranking features aggregate a broad training window. Although validation is
  strictly later, a stronger experiment would use an inner chronological
  feature cutoff and label window.
- Fairness across demographic groups cannot be measured because MovieLens 25M
  does not provide the required attributes in this pipeline.
- Live TensorFlow SavedModel loading in the pinned Triton container remains an
  explicit acceptance-test gap.

## Privacy and safety

The public dataset uses numeric user IDs, but interaction histories can still
be sensitive when applied to real systems. A production deployment would need
data minimization, retention/deletion policies, access controls, encryption,
consent and regional compliance review. Do not log raw personal identifiers or
commit event data.

## Reproducibility

Use `configs/dev.toml` for the documented bounded experiment and
`configs/full.toml` for an uncapped full-data run. Seeds and limits are stored
in configuration, while generated reports and immutable run identifiers record
the resulting evidence. Hardware, dependency versions, and configuration must
be reported with any comparison.
