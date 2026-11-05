import pytest

from recommender.serving.experiment import (
    InMemoryExperimentStore,
    assign_variant,
)


def test_assignment_is_sticky_and_reasonably_balanced() -> None:
    assignments = [
        assign_variant("ranker-v1", f"user-{index}") for index in range(1000)
    ]

    assert assign_variant("ranker-v1", "user-7") == assign_variant(
        "ranker-v1", "user-7"
    )
    wide_count = assignments.count("wide_and_deep")
    assert 450 <= wide_count <= 550


def test_event_store_validates_click_and_calculates_ctr() -> None:
    store = InMemoryExperimentStore()
    first = store.record_impression(
        experiment_id="exp",
        variant="dcn",
        subject_id="user-1",
        movie_ids=(10, 20),
        model_version="v1",
    )
    store.record_impression(
        experiment_id="exp",
        variant="dcn",
        subject_id="user-2",
        movie_ids=(20, 30),
        model_version="v1",
    )
    click = store.record_click(first.impression_id, 20)

    assert click.rank == 2
    metrics = store.metrics("exp")["variants"]["dcn"]
    assert metrics["impressions"] == 2
    assert metrics["clicks"] == 1
    assert metrics["ctr"] == 0.5
    assert metrics["discounted_clicks_per_impression"] == pytest.approx(
        1 / 3.169925001442312
    )

    store.record_click(first.impression_id, 10)
    assert store.metrics("exp")["variants"]["dcn"]["ctr"] == 0.5


def test_click_must_reference_an_impressed_movie() -> None:
    store = InMemoryExperimentStore()
    impression = store.record_impression(
        experiment_id="exp",
        variant="wide_and_deep",
        subject_id="user-1",
        movie_ids=(10,),
        model_version="v1",
    )

    with pytest.raises(ValueError, match="not present"):
        store.record_click(impression.impression_id, 99)
