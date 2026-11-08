# Local Infrastructure Contract

`compose.yaml` provides a reproducible local API and supporting infrastructure,
not a production cluster.

## Services

| Service | Pinned image | Host ports | Persistence |
|---|---|---|---|
| FastAPI demo | Built from the repository `Dockerfile` | 8000 | Read-only generated artifacts and processed features |
| Redis | `redis:8.2.10-alpine` | 6379 | Named volume with AOF and periodic RDB |
| Triton | `nvcr.io/nvidia/tritonserver:25.02-py3` | 8001 HTTP, 8002 gRPC, 8003 metrics | Generated read-only model repository |
| Airflow utility | `apache/airflow:3.3.2-python3.11` | none | Ignored `.airflow/` SQLite metadata for local DAG tests |

Redis and Triton have health checks. Airflow is behind the `orchestration`
profile and is invoked as a one-shot utility for database migration and
`airflow dags test`; it is not presented as a production scheduler deployment.
The FastAPI image is behind the `api` profile and defaults to local SavedModel
inference. It requires the generated model, index, and feature artifacts.

## Commands

```bash
make infra-up
make test-redis
make verify-triton
make infra-down
```

To build and run the API with Redis after generating local artifacts:

```bash
make demo-up
make demo-down
```

`make infra-up` regenerates the model repository before starting services.
Redis failures remain fail-open for recommendations. Triton failures are
fail-closed at API startup when the Triton backend is selected, because serving
a silently different model path would invalidate the experiment.

The real Redis integration test writes a unique namespace, verifies JSON cache
TTL, appends two impressions and one validated click to a Stream, checks CTR
and stream length, and removes only its exact test keys afterward.
