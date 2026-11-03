from pathlib import Path

import pytest

from recommender.config import FeatureConfig, load_config


def test_load_config_reads_expected_values() -> None:
    config = load_config(Path("configs/base.toml"))

    assert config.data.raw_dir == Path("data/raw/ml-25m")
    assert config.split.train_fraction == 0.8
    assert config.baseline.recommendation_sizes == (10, 20)
    assert config.features.recent_window_days == 90
    assert config.features.popularity_buckets == 10
    assert config.retrieval.embedding_dimension == 64
    assert config.retrieval.evaluation_ks == (10, 50, 100, 200)


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
