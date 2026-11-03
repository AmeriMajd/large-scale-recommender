import pytest

from recommender.retrieval.fallbacks import (
    RetrievalStrategy,
    choose_retrieval_strategy,
)


@pytest.mark.parametrize(
    ("is_new_user", "is_new_item", "expected"),
    [
        (False, False, RetrievalStrategy.TWO_TOWER),
        (True, False, RetrievalStrategy.POPULARITY),
        (False, True, RetrievalStrategy.CONTENT),
        (True, True, RetrievalStrategy.CONTENT_POPULARITY),
    ],
)
def test_choose_retrieval_strategy_covers_every_cold_start_state(
    is_new_user: bool,
    is_new_item: bool,
    expected: RetrievalStrategy,
) -> None:
    assert (
        choose_retrieval_strategy(
            is_new_user=is_new_user,
            is_new_item=is_new_item,
        )
        == expected
    )
