"""FastAPI interface for recommendations, feedback, and A/B metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from recommender.config import load_config


class EngineProtocol(Protocol):
    experiments: Any

    def recommend(self, user_id: int, limit: int | None = None) -> dict: ...

    def recommend_cold_start(
        self,
        *,
        subject: str,
        selected_genres: tuple[str, ...] = (),
        liked_movie_ids: tuple[int, ...] = (),
        limit: int | None = None,
    ) -> dict: ...

    def search_movies(self, query: str, limit: int = 8) -> list[dict]: ...

    def health(self) -> dict: ...


class ColdStartRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    selected_genres: list[str] = Field(default_factory=list, max_length=20)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)


class ImpressionRequest(BaseModel):
    experiment_id: str
    variant: str
    subject_id: str
    movie_ids: list[int] = Field(min_length=1, max_length=100)
    model_version: str


class ClickRequest(BaseModel):
    impression_id: str
    movie_id: int


def create_app(engine: EngineProtocol) -> FastAPI:
    """Create an API with its heavy artifact-loaded engine injected."""
    app = FastAPI(
        title="Large-Scale Recommendation Engine",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        return engine.health()

    @app.get("/recommendations/{user_id}")
    def recommendations(
        user_id: int,
        limit: int = Query(default=10, ge=1, le=100),
    ) -> dict:
        return engine.recommend(user_id, limit)

    @app.post("/recommendations/cold-start")
    def cold_start(request: ColdStartRequest) -> dict:
        return engine.recommend_cold_start(
            subject=f"session-{request.session_id}",
            selected_genres=tuple(request.selected_genres),
            liked_movie_ids=tuple(request.liked_movie_ids),
            limit=request.limit,
        )

    @app.get("/catalog/search")
    def catalog_search(
        q: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=8, ge=1, le=20),
    ) -> dict:
        return {"results": engine.search_movies(q, limit)}

    @app.post("/events/impression", status_code=201)
    def impression(request: ImpressionRequest) -> dict:
        value = engine.experiments.record_impression(
            experiment_id=request.experiment_id,
            variant=request.variant,
            subject_id=request.subject_id,
            movie_ids=tuple(request.movie_ids),
            model_version=request.model_version,
        )
        return {"impression_id": value.impression_id}

    @app.post("/events/click", status_code=201)
    def click(request: ClickRequest) -> dict:
        try:
            value = engine.experiments.record_click(
                request.impression_id, request.movie_id
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "impression_id": value.impression_id,
            "movie_id": value.movie_id,
            "rank": value.rank,
        }

    @app.get("/experiments/{experiment_id}/metrics")
    def experiment_metrics(experiment_id: str) -> dict:
        return engine.experiments.metrics(experiment_id)

    ui_dir = Path(__file__).resolve().parents[3] / "app"
    if ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app


def main() -> None:
    import uvicorn

    from recommender.serving.engine import build_default_engine

    config = load_config()
    app = create_app(build_default_engine(config))
    uvicorn.run(
        app,
        host=config.serving.api_host,
        port=config.serving.api_port,
    )


if __name__ == "__main__":
    main()
