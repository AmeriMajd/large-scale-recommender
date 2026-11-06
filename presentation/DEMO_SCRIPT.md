# Final Presentation and Demo Script

Target length: 9–10 minutes total.

- Slides 1–9: about 5 minutes
- Live demo: about 4 minutes
- Conclusion: 30–45 seconds

## Before the presentation

Open two terminals from the repository root.

Terminal 1 — start the recommendation service:

```bash
make infra-up
make serve
```

Terminal 2 — serve the presentation:

```bash
python3 -m http.server 4173 -d presentation
```

Open these tabs before screen sharing:

1. Presentation: `http://127.0.0.1:4173`
2. Model Observatory: `http://127.0.0.1:8000`
3. API documentation, optional: `http://127.0.0.1:8000/docs`

Preflight checks:

```bash
curl -fsS http://127.0.0.1:8000/health
make validate-pipeline
```

Keep the presentation tab first and the Model Observatory tab second. Do not run a new multi-hour training job during the demo.

## Slide script

### Slide 1 — Opening thesis

“This project is a production-style personalized recommendation system built on MovieLens 25M. The main result is not one model score. It is the complete lifecycle: leakage-safe data, two-stage retrieval and ranking, vector search, online serving, feedback, and Airflow orchestration.”

### Slide 2 — Temporal data boundary

“The first production decision is the data boundary. I split ratings chronologically: about 20 million interactions for training, 2.5 million for validation, and 2.5 million for final test. Every learned feature is frozen at the training cutoff, so future interactions cannot leak into training.”

### Slide 3 — Architecture

“The system has two online stages. Spark builds the training and feature artifacts. A two-tower model retrieves roughly 200 candidates through FAISS. Wide & Deep or DCN then re-ranks those candidates using richer context. FastAPI serves the results, Redis provides versioned caching and event storage, and Airflow orchestrates the existing commands.”

### Slide 4 — Retrieval scale

“The learned retrieval vocabulary contains 137,622 users and 24,728 candidate movies. The exported item matrix has 64-dimensional embeddings. At request time, the user tower produces one vector, FAISS searches the matrix, and only the top 200 items continue to the expensive ranking stage.”

### Slide 5 — FAISS benchmark

“At this catalog size, high-recall HNSW is roughly tied with exact search. With efSearch 256 it preserves 98.59 percent of exact top-200 results, but its latency is similar to exact search. I report that result honestly: the index contract is ready for larger catalogs, but this dataset is not large enough to show a dramatic high-recall speedup.”

### Slide 6 — Ranking comparison

“Re-ranking improves the order of the retrieved candidates. DCN achieved the best validation NDCG at 0.0753, narrowly ahead of Wide & Deep at 0.0740. This is a small but reproducible improvement, selected using future validation data and early stopping.”

### Slide 7 — Serving

“The first known-user request performs the full tower, FAISS, and DCN path. It measured 374 milliseconds on the development CPU. Warm calls measured 13.79 milliseconds at P50. That warm benchmark used the deterministic in-memory fallback, so I do not present it as a Redis production-load result. Redis failures are intentionally fail-open.”

### Slide 8 — Airflow

“Airflow turns the reproducible commands into ten dependency-aware tasks. Each task validates concrete outputs before downstream work can start. The complete DAG ran successfully in the pinned Airflow 3.3.2 container on 5 November 2026 and checked fourteen artifacts. The demo uses validate mode, which proves orchestration and the existing artifact chain—not a fresh multi-hour retrain.”

### Slide 9 — Validation status

“The core system has concrete evidence across Spark, retrieval, FAISS, both rankers, the API, Redis, the UI, and Airflow. The latest suite reports 58 passing tests and two service-gated skips. Triton repository generation and client integration are implemented and unit-tested, but the 10.8-gigabyte live Triton server run remains optional and is not claimed as completed.”

## Live demo — 4 minutes

Switch to `http://127.0.0.1:8000`.

### 1. Known-user inference — 60 seconds

Action:

1. Keep user ID `42`.
2. Click **RUN INFERENCE**.
3. Point to the health ribbon, selected variant, and dense result table.

Say:

“The service reports the live catalog size, cache health, and inference backend. This request is assigned to a ranking variant. The table exposes both ranker and retrieval scores so the final order is inspectable.”

Expected evidence:

- approximately 137.6K known users
- approximately 24.7K indexed items
- local TensorFlow ready
- ten ranked results

### 2. Request trace and feedback — 55 seconds

Action:

1. Point to the request trace.
2. Click the first result row.
3. Point to the selected-result inspector and feedback confirmation.

Say:

“The trace shows the full online path: user tower, FAISS retrieval of 200 candidates, the assigned ranker, and the top-ten response. Clicking a row records one validated feedback event tied to the impression and rank.”

### 3. Cold start — 65 seconds

Action:

1. Open **Cold start**.
2. Select `Animation` and `Sci-Fi`.
3. Search for `Toy Story` and add it.
4. Click **GENERATE TOP 10**.

Say:

“A new session has no learned user embedding, so the system routes to a content-and-popularity fallback. Genres and liked titles form a temporary profile. The response keeps the same versioned API contract and remains cacheable.”

If catalog search is slow, skip adding a title and generate from genres only.

### 4. A/B metrics — 45 seconds

Action:

1. Open **A/B metrics**.
2. Point to impressions, clicks, and CTR by variant.

Say:

“Experiment assignment is deterministic for a subject. Impressions are recorded with the response, and clicks update the observed demo metrics. These are demo interactions only, not MovieLens historical labels.”

### 5. Airflow proof — 15 seconds

Return to the presentation and show slide 8 or the committed Day 10 report.

Say:

“For time, I am showing the recorded successful Airflow run rather than re-running all ten container tasks live. The reproducible commands are `make validate-pipeline` and `make airflow-test`.”

## Conclusion

Return to slide 12.

“The final result is a reproducible recommendation system, not a notebook prototype. Its data boundaries are explicit, its model and latency trade-offs are measured, and its serving and orchestration paths are testable. The next optional infrastructure proof is live Triton equivalence, followed by production-scale concurrency and load testing.”

## Failure-safe fallback

If the API does not start:

1. Keep the presentation running.
2. Use slide 10’s real captured interface.
3. Narrate the four demo steps without claiming a live result.
4. Show the committed contracts and test results.

If Redis fails:

- Continue the demo.
- Say: “The service is using its fail-open in-memory backend; this behavior is intentional.”

If Airflow cannot start:

- Show slide 8 and `docs/DAY_10_REPORT.md`.
- State the recorded date and validate-mode scope.

Never claim:

- a fresh 25M retrain happened during the demo;
- warm in-memory latency is a Redis load benchmark;
- Triton numerical equivalence has been validated live;
- demo A/B clicks are offline MovieLens metrics.

## Likely questions

**Why does the two-tower model underperform popularity?**

The development retrieval model is ID-only and trained on a coverage-preserving sample. Popularity is a strong baseline for sparse implicit feedback. The result is reported honestly; stronger towers would add sequence, genre, and richer behavioral features.

**Why keep two-stage retrieval if popularity is stronger?**

The project demonstrates the scalable system boundary. Retrieval constrains expensive ranking to a small candidate set, while the ranker adds richer context. The architecture supports better future retrieval models without changing serving contracts.

**Why use exact FAISS when HNSW exists?**

At 24.7K candidates, exact search is already fast and provides perfect top-200 recall. HNSW is persisted and benchmarked, but exact search is the sensible current default until catalog scale changes the trade-off.

**Why DCN over Wide & Deep?**

DCN produced the best future-validation NDCG@10 under the same candidates and evaluation users. The difference is small, so both models remain available for deterministic online experimentation.

**What exactly did Airflow validate?**

It parsed the real DAG, resolved dependencies, templated and executed all ten tasks in validate mode, and checked fourteen existing artifacts. It did not retrain the complete 25M pipeline during the short run.

**What remains for Triton?**

The model repository, tensor contracts, exporter, client adapter, and unit tests exist. The remaining optional step is running the 10.8 GB TensorFlow-capable server image and measuring live numerical equivalence and HTTP latency.
