FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    RECSYS_API_HOST=0.0.0.0

WORKDIR /opt/recommender

RUN groupadd --system recommender \
    && useradd --system --gid recommender --home-dir /opt/recommender recommender

COPY requirements-api.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs

RUN python -m pip install --no-cache-dir -r requirements-api.lock \
    && python -m pip install --no-cache-dir --no-deps ".[retrieval,indexing,serving]"

USER recommender
EXPOSE 8000

CMD ["recsys-serve"]
