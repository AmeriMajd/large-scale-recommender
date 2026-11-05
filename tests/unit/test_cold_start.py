from recommender.serving.cold_start import CatalogItem, ColdStartRecommender


def _catalog() -> list[CatalogItem]:
    return [
        CatalogItem(1, "Action One", ("Action",), 1.0, 4.5, 100),
        CatalogItem(2, "Action Two", ("Action",), 0.9, 4.0, 90),
        CatalogItem(3, "Action Three", ("Action",), 0.8, 4.0, 80),
        CatalogItem(4, "Action Four", ("Action",), 0.7, 4.0, 70),
        CatalogItem(5, "Drama One", ("Drama",), 0.6, 5.0, 60),
        CatalogItem(6, "Comedy One", ("Comedy",), 0.5, 5.0, 50),
    ]


def test_popularity_fallback_applies_genre_diversity() -> None:
    results = ColdStartRecommender(_catalog(), max_per_genre=2).recommend(limit=4)

    assert [result.movie_id for result in results[:2]] == [1, 2]
    assert {result.movie_id for result in results} == {1, 2, 5, 6}
    assert {result.reason for result in results} == {"diversified_popularity"}


def test_content_profile_uses_genres_and_excludes_liked_movies() -> None:
    results = ColdStartRecommender(_catalog()).recommend(
        selected_genres=("action",), liked_movie_ids=(2,), limit=3
    )

    assert results[0].movie_id == 1
    assert 2 not in {result.movie_id for result in results}
    assert all(result.reason == "genre_content_popularity" for result in results)


def test_unknown_preferences_fall_back_to_popularity() -> None:
    results = ColdStartRecommender(_catalog()).recommend(
        selected_genres=("Not-A-Genre",), liked_movie_ids=(999,), limit=2
    )

    assert [result.movie_id for result in results] == [1, 2]
