import os

import pytest

from recommender.serving.verify_triton import verify

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TRITON_INTEGRATION") != "1",
    reason="set RUN_TRITON_INTEGRATION=1 with Triton running",
)


def test_live_triton_matches_local_savedmodels() -> None:
    report = verify(repetitions=2)

    assert report["triton_health"]["ready"]
    assert report["user_embedding_max_abs_error"] < 1e-5
    assert max(report["ranker_max_abs_error"].values()) < 1e-5
