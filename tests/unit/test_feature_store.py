from recommender.serving.engine import FeatureStore


def test_catalog_search_prefers_prefix_then_popularity() -> None:
    store = FeatureStore.__new__(FeatureStore)
    store.items = {
        1: {
            "movieId": 1,
            "title": "A Toy Story",
            "genre_list": ["Animation"],
            "popularity_percentile": 0.0,
        },
        2: {
            "movieId": 2,
            "title": "Toy Soldiers",
            "genre_list": ["Action"],
            "popularity_percentile": 0.8,
        },
        3: {
            "movieId": 3,
            "title": "Toy Story",
            "genre_list": ["Animation", "Comedy"],
            "popularity_percentile": 0.1,
        },
    }

    results = store.search("toy", limit=3)

    assert [item["movie_id"] for item in results] == [3, 2, 1]
    assert results[0]["genres"] == ["Animation", "Comedy"]
