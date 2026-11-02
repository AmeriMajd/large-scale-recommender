from pathlib import Path

import pytest

from recommender.config import load_config


def test_load_config_reads_expected_day_one_values() -> None:
    config = load_config(Path("configs/base.toml"))

    assert config.data.raw_dir == Path("data/raw/ml-25m")
    assert config.split.train_fraction == 0.8
    assert config.baseline.recommendation_sizes == (10, 20)


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
