from recommender.baselines.popularity import recommend_popular


def test_popularity_recommendations_exclude_seen_items() -> None:
    assert recommend_popular([10, 20, 30, 40], {10, 30}, 2) == [20, 40]


def test_popularity_recommendations_can_return_short_catalog() -> None:
    assert recommend_popular([10, 20], {10}, 5) == [20]
