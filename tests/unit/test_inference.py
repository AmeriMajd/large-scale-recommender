import numpy as np
import pytest

from recommender.serving.inference import TritonHttpInference


class FakeResponse:
    def __init__(self, payload=None, status_code=200) -> None:
        self.payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.requests = []

    def post(self, path: str, json: dict) -> FakeResponse:
        self.requests.append((path, json))
        if "user_tower" in path:
            return FakeResponse(
                {
                    "outputs": [
                        {
                            "name": "embedding",
                            "shape": [1, 64],
                            "datatype": "FP32",
                            "data": [0.125] * 64,
                        }
                    ]
                }
            )
        rows = json["inputs"][0]["shape"][0]
        return FakeResponse(
            {
                "outputs": [
                    {
                        "name": "logit",
                        "shape": [rows],
                        "datatype": "FP32",
                        "data": list(np.arange(rows, dtype=np.float32)),
                    }
                ]
            }
        )

    def get(self, path: str) -> FakeResponse:
        self.requests.append((path, None))
        return FakeResponse()


def test_triton_adapter_serializes_ids_and_dense_features() -> None:
    client = FakeClient()
    backend = TritonHttpInference("http://triton", client=client)

    embedding = backend.user_embedding(42)
    logits = backend.ranker_logits(
        "dcn",
        {
            "user_id": np.asarray([42, 42], dtype=np.int64),
            "retrieval_score": np.asarray([0.2, 0.1], dtype=np.float32),
        },
    )

    assert embedding.shape == (1, 64)
    assert logits.tolist() == [0.0, 1.0]
    user_input = client.requests[0][1]["inputs"][0]
    assert user_input == {
        "name": "user_id",
        "shape": [1],
        "datatype": "INT64",
        "data": [42],
    }
    assert backend.health()["ready"]


def test_triton_adapter_rejects_unknown_ranker() -> None:
    backend = TritonHttpInference("http://triton", client=FakeClient())

    with pytest.raises(ValueError, match="unknown ranker"):
        backend.ranker_logits("other", {"user_id": np.asarray([1])})
