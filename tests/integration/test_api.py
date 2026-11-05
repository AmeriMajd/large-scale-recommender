from fastapi.testclient import TestClient

from recommender.serving.api import create_app
from recommender.serving.experiment import InMemoryExperimentStore


class FakeEngine:
    def __init__(self) -> None:
        self.experiments = InMemoryExperimentStore()

    def health(self) -> dict:
        return {"status": "ok", "cache_healthy": True}

    def recommend(self, user_id: int, limit: int | None = None) -> dict:
        return {
            "subject": f"user-{user_id}",
            "variant": "dcn",
            "recommendations": [{"movie_id": 10}] * (limit or 10),
        }

    def recommend_cold_start(
        self,
        *,
        subject: str,
        selected_genres: tuple[str, ...] = (),
        liked_movie_ids: tuple[int, ...] = (),
        limit: int | None = None,
    ) -> dict:
        return {
            "subject": subject,
            "variant": "cold_start",
            "strategy": (
                "genre_content_popularity"
                if selected_genres or liked_movie_ids
                else "diversified_popularity"
            ),
            "recommendations": [{"movie_id": 20}] * (limit or 10),
        }


def test_health_known_and_cold_start_endpoints() -> None:
    client = TestClient(create_app(FakeEngine()))

    assert client.get("/health").json()["status"] == "ok"
    known = client.get("/recommendations/42?limit=3")
    assert known.status_code == 200
    assert len(known.json()["recommendations"]) == 3
    cold = client.post(
        "/recommendations/cold-start",
        json={
            "session_id": "abc",
            "selected_genres": ["Action"],
            "liked_movie_ids": [10],
            "limit": 2,
        },
    )
    assert cold.status_code == 200
    assert cold.json()["strategy"] == "genre_content_popularity"


def test_impression_click_and_metrics_round_trip() -> None:
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    impression = client.post(
        "/events/impression",
        json={
            "experiment_id": "exp",
            "variant": "dcn",
            "subject_id": "user-1",
            "movie_ids": [10, 20],
            "model_version": "v1",
        },
    )
    impression_id = impression.json()["impression_id"]
    click = client.post(
        "/events/click",
        json={"impression_id": impression_id, "movie_id": 20},
    )
    metrics = client.get("/experiments/exp/metrics")

    assert impression.status_code == 201
    assert click.status_code == 201
    assert click.json()["rank"] == 2
    assert metrics.json()["variants"]["dcn"]["ctr"] == 1.0


def test_invalid_click_and_limit_return_client_errors() -> None:
    client = TestClient(create_app(FakeEngine()))

    assert client.post(
        "/events/click",
        json={"impression_id": "missing", "movie_id": 1},
    ).status_code == 404
    assert client.get("/recommendations/1?limit=0").status_code == 422
