from pathlib import Path

import pytest

from recommender.config import (
    FeatureConfig,
    IndexingConfig,
    RankingConfig,
    ServingConfig,
    load_config,
)


def test_load_config_reads_expected_values() -> None:
    config = load_config(Path("configs/base.toml"))

    assert config.data.raw_dir == Path("data/raw/ml-25m")
    assert config.split.train_fraction == 0.8
    assert config.baseline.recommendation_sizes == (10, 20)
    assert config.features.recent_window_days == 90
    assert config.features.popularity_buckets == 10
    assert config.retrieval.embedding_dimension == 64
    assert config.retrieval.evaluation_ks == (10, 50, 100, 200)
    assert config.indexing.k == 200
    assert config.indexing.hnsw_ef_search_values == (64, 128, 256, 512)
    assert config.ranking.candidate_k == 200
    assert config.ranking.deep_layer_sizes == (128, 64)
    assert config.serving.cache_ttl_seconds == 900
    assert config.serving.experiment_id == "ranking-model-v1"


def test_config_profiles_inherit_base_and_full_removes_sampling_cap() -> None:
    development = load_config(Path("configs/dev.toml"))
    full = load_config(Path("configs/full.toml"))

    assert development.retrieval.max_train_interactions == 2_000_000
    assert development.retrieval.sampling_buckets == 1
    assert development.retrieval.sampling_modulus == 10
    assert full.retrieval.max_train_interactions == 19_986_521
    assert full.retrieval.sampling_buckets == full.retrieval.sampling_modulus == 1
    assert full.ranking.max_training_positives == 19_986_521
    assert full.spark.driver_memory == "12g"


@pytest.mark.parametrize(
    ("recent_window_days", "popularity_buckets", "message"),
    [(0, 10, "recent_window_days"), (90, 1, "popularity_buckets")],
)
def test_feature_config_rejects_invalid_values(
    recent_window_days: int,
    popularity_buckets: int,
    message: str,
) -> None:
    config = FeatureConfig(recent_window_days, popularity_buckets)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[data]
raw_dir = "raw"
processed_dir = "processed"
reports_dir = "reports"
[split]
train_fraction = 0.8
validation_fraction = 0.2
test_fraction = 0.2
quantile_error = 0.0
[baseline]
positive_rating = 4.0
recommendation_sizes = [10]
[features]
recent_window_days = 90
popularity_buckets = 10
[spark]
app_name = "test"
master = "local[1]"
shuffle_partitions = 1
driver_memory = "1g"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(config_path)


def test_indexing_config_requires_ordered_search_values() -> None:
    config = IndexingConfig(
        index_dir=Path("indexes"),
        query_count=10,
        k=5,
        warmup_queries=1,
        benchmark_repeats=1,
        omp_threads=1,
        hnsw_m=16,
        hnsw_ef_construction=40,
        hnsw_ef_search_values=(32, 16),
        random_seed=42,
    )

    with pytest.raises(ValueError, match="unique and increasing"):
        config.validate()


def test_ranking_metric_k_cannot_exceed_candidate_count() -> None:
    config = RankingConfig(
        model_dir=Path("models"),
        max_training_positives=10,
        negatives_per_positive=2,
        candidate_k=10,
        max_validation_users=5,
        parquet_batch_size=10,
        batch_size=4,
        epochs=1,
        early_stopping_patience=0,
        learning_rate=0.001,
        id_embedding_dimension=4,
        deep_layer_sizes=(8,),
        cross_layers=1,
        evaluation_ks=(5, 20),
        random_seed=42,
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        config.validate()


def test_serving_candidate_count_must_cover_response_limit() -> None:
    config = ServingConfig(
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=60,
        event_stream_maxlen=100,
        recommendation_limit=20,
        retrieval_candidate_k=10,
        experiment_id="experiment",
        model_namespace="recs",
        api_host="127.0.0.1",
        api_port=8000,
        inference_backend="local",
        triton_http_url="http://127.0.0.1:8001",
        triton_timeout_seconds=5.0,
    )

    with pytest.raises(ValueError, match="cannot be smaller"):
        config.validate()
