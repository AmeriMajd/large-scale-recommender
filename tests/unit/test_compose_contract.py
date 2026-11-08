from pathlib import Path


def test_compose_pins_services_and_health_checks() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "redis:8.2.10-alpine" in compose
    assert "nvcr.io/nvidia/tritonserver:25.02-py3" in compose
    assert "apache/airflow:3.3.2-python3.11" in compose
    assert compose.count("healthcheck:") >= 3
    assert "redis-data:/data" in compose
    assert "./triton-models:/models:ro" in compose
    assert 'profiles: ["api"]' in compose
    assert "./artifacts:/opt/recommender/artifacts:ro" in compose
