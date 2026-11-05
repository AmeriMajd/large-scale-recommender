# Day 9 Report: Docker Compose and Real Redis

## Outcome

Day 9 replaces the cache/event-store assumption with a tested Redis service.
Docker Compose defines pinned Redis and Triton containers, gives Redis a named
volume, enables append-only persistence plus periodic snapshots, maps explicit
ports, and declares health checks. Airflow is included behind a separate
profile for Day 10 rather than starting an unused scheduler during normal API
work. Redis was exercised live; the much larger Triton image and its live test
remain explicitly pending.

The real Redis integration test passed against `redis:8.2.10-alpine`. It wrote
and read a cached recommendation object, observed a positive TTL, recorded two
impressions and one rank-two click, calculated CTR 0.5, and verified three
Stream entries. Redis reported healthy with AOF enabled. A controlled container
restart preserved a test value through the named volume; the value was then
deleted.

## Concepts learned

Container state and data state are different. A container may be replaced; the
named volume owns Redis durability. AOF records writes incrementally while RDB
creates point-in-time snapshots. This local combination is useful for a demo,
but it is not replication, backup, or high availability.

Health checks answer whether a dependency is ready, not merely whether its
process exists. Compose reports Redis healthy only after `redis-cli ping`
succeeds. The API still treats caching as optional: Redis failure increases
work but must not block recommendations.

Integration tests should have isolated ownership. The Redis test uses a UUID
namespace and deletes only its exact cache, stream, and impression keys. It
never flushes the database.

## Key code walkthrough

[`compose.yaml`](../compose.yaml) pins all service images, publishes Redis and
Triton ports, mounts models read-only, persists Redis data, and keeps Airflow in
the `orchestration` profile.

[`test_redis.py`](../tests/integration/test_redis.py) exercises the real
`RedisKeyValueBackend` and `RedisExperimentStore`, including TTL, click
validation, Stream length, and experiment aggregation.

[`test_compose_contract.py`](../tests/unit/test_compose_contract.py) prevents
accidental floating image tags, missing health checks, or loss of the model and
data mounts.

## Complete debugging log

| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| Docker commands initially returned permission denied on `/var/run/docker.sock` | The restricted execution environment does not expose the daemon by default | Ran the explicitly requested Docker workflow with daemon permission | CLI availability does not prove daemon access |
| Starting Redis and Triton together made both downloads progress slowly | Two independent image pulls competed for limited registry bandwidth | Stopped the combined pull, completed and tested Redis first, then cancelled the disproportionate 10.8 GB Triton pull | Pull large infrastructure images separately on constrained links and state pending live checks clearly |
| Redis had been unavailable in the Day 7 benchmark | No service lifecycle existed in the repository | Added pinned Compose service, health check, volume, and integration target | A client adapter is not an integration until a real server is exercised |
| Cache tests alone could not prove persistence | The in-memory backend intentionally disappears with the process | Stored a marker, forced a Redis save, restarted the container, read it back, then deleted it | Test the failure/restart behavior claimed in documentation |
| A broad cleanup would risk deleting developer data | Redis integration creates several related keys | Generated a unique namespace and deleted exact owned keys in `finally` | Never use `FLUSHDB` in shared integration tests |

## Tests and evidence

- Docker Compose configuration validates successfully.
- The pinned-image/health/mount unit contract passes.
- Real Redis cache and Streams integration: one test passed in 0.09 seconds.
- Redis container status: healthy on port 6379.
- Persistence inspection: `aof_enabled:1`, successful RDB save.
- Controlled restart: stored value returned `ok` after container restart.

This does not test Redis replication, Sentinel/Cluster failover, authentication,
TLS, eviction pressure, or concurrent cache stampedes.

## Learning resources, in order

1. [Docker Compose services](https://docs.docker.com/reference/compose-file/services/)
   — focus on health checks, volumes, restart policies, and profiles.
2. [Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)
   — compare RDB, AOF, durability, and rewrite behavior.
3. [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams/)
   — revisit `XADD`, trimming, consumer groups, and delivery semantics.
4. [Docker volumes](https://docs.docker.com/engine/storage/volumes/)
   — understand why a named volume outlives a replaced container.

## Practice exercises

1. Explain why `restart: unless-stopped` is not high availability.
2. What data can be lost with periodic RDB snapshots but no AOF?
3. Add Redis authentication without committing the password.
4. Simulate Redis downtime and predict API health, latency, and correctness.
5. Why does the integration test delete exact keys instead of calling
   `FLUSHDB`?

## Reproduction

```bash
make infra-up
make test-redis
docker compose ps redis
docker compose exec redis redis-cli INFO persistence
```

See [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) for the complete service contract.
