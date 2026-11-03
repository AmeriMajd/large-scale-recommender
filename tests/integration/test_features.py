from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, SparkSession

from recommender.config import load_config
from recommender.features.build import (
    SECONDS_PER_DAY,
    build_feature_tables,
    build_item_features,
    build_user_features,
    profile_cold_start_context,
)

CUTOFF = 1_466_573_687  # 22 June 2016 UTC


def _ratings(spark: SparkSession, rows: list[tuple]) -> DataFrame:
    return spark.createDataFrame(
        rows,
        schema="userId long, movieId long, rating double, timestamp long",
    )


def _movies(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (1, "First Movie (2000)", "Action|Drama", ["Action", "Drama"], 2000),
            (2, "Second Movie (2010)", "Drama", ["Drama"], 2010),
            (3, "Future Movie (2018)", "Comedy", ["Comedy"], 2018),
            (4, "Unknown Year", "", [], None),
        ],
        schema=(
            "movieId long, title string, genres string, "
            "genre_list array<string>, release_year integer"
        ),
    )


def _training(spark: SparkSession) -> DataFrame:
    return _ratings(
        spark,
        [
            (1, 1, 5.0, CUTOFF - 100 * SECONDS_PER_DAY),
            (1, 2, 3.0, CUTOFF - 10 * SECONDS_PER_DAY),
            (2, 1, 4.0, CUTOFF - 5 * SECONDS_PER_DAY),
        ],
    )


def test_user_and_item_features_have_expected_training_statistics(
    spark: SparkSession,
) -> None:
    train = _training(spark)

    users = {
        row["userId"]: row.asDict()
        for row in build_user_features(train, CUTOFF, 4.0, 90).collect()
    }
    user_one = users[1]
    assert user_one["interaction_count"] == 2
    assert user_one["distinct_movie_count"] == 2
    assert user_one["mean_rating"] == pytest.approx(4.0)
    assert user_one["positive_rate"] == pytest.approx(0.5)
    assert user_one["recent_interaction_count"] == 1
    assert user_one["activity_span_days"] == 90
    assert user_one["days_since_last_interaction"] == 10
    assert user_one["feature_cutoff_epoch"] == CUTOFF

    items = {
        row["movieId"]: row.asDict()
        for row in build_item_features(train, _movies(spark), CUTOFF, 4.0, 10).collect()
    }
    assert items[1]["interaction_count"] == 2
    assert items[1]["mean_rating"] == pytest.approx(4.5)
    assert items[1]["popularity_rank"] == 1
    assert items[3]["interaction_count"] == 0
    assert items[3]["mean_rating"] is None
    assert items[3]["is_seen_in_train"] is False
    assert items[3]["is_future_release_at_cutoff"] is True
    assert items[4]["release_year_missing"] is True
    assert items[4]["is_future_release_at_cutoff"] is None


def test_feature_tables_are_leakage_safe_and_mark_cold_start(
    spark: SparkSession,
) -> None:
    train = _training(spark)
    validation = _ratings(
        spark,
        [
            (1, 1, 0.5, CUTOFF + 1),
            (1, 3, 5.0, CUTOFF + 2),
            (3, 1, 5.0, CUTOFF + 3),
            (3, 3, 5.0, CUTOFF + 4),
        ],
    )
    test = _ratings(spark, [(4, 4, 5.0, CUTOFF + 5)])

    tables = build_feature_tables(
        train,
        validation,
        test,
        _movies(spark),
        load_config("configs/base.toml"),
        CUTOFF,
    )

    users = {row["userId"]: row.asDict() for row in tables.user_features.collect()}
    assert set(users) == {1, 2}
    assert users[1]["mean_rating"] == pytest.approx(4.0)

    items = {row["movieId"]: row.asDict() for row in tables.item_features.collect()}
    assert items[3]["interaction_count"] == 0

    genres = {
        (row["userId"], row["genre"]): row.asDict()
        for row in tables.user_genre_features.collect()
    }
    assert genres[(1, "Action")]["genre_interaction_share"] == pytest.approx(1 / 3)
    assert genres[(1, "Drama")]["genre_interaction_share"] == pytest.approx(2 / 3)
    assert genres[(1, "Action")]["recent_genre_share"] == pytest.approx(0.0)
    assert genres[(1, "Drama")]["recent_genre_share"] == pytest.approx(1.0)

    cold_types = {
        (row["userId"], row["movieId"]): row["cold_start_type"]
        for row in tables.validation_context.collect()
    }
    assert cold_types == {
        (1, 1): "known_user_and_item",
        (1, 3): "new_item",
        (3, 1): "new_user",
        (3, 3): "new_user_and_item",
    }
    cold_profile = profile_cold_start_context(tables.validation_context)
    assert cold_profile == {
        "interactions": 4,
        "distinct_users": 2,
        "distinct_items": 2,
        "new_users": 1,
        "new_items": 1,
        "interactions_by_type": {
            "known_user_and_item": 1,
            "new_item": 1,
            "new_user": 1,
            "new_user_and_item": 1,
        },
    }
    test_row = tables.test_context.first()
    assert test_row["cold_start_type"] == "new_user_and_item"
