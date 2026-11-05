"""Interchangeable local TensorFlow and Triton V2 inference adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

RANKER_NAMES = frozenset({"wide_and_deep", "dcn"})


class InferenceBackend(Protocol):
    name: str
    ranker_names: frozenset[str]

    def user_embedding(self, user_id: int) -> np.ndarray: ...

    def ranker_logits(
        self, variant: str, features: dict[str, Any]
    ) -> np.ndarray: ...

    def health(self) -> dict: ...


class LocalTensorFlowInference:
    """Call reloaded TensorFlow SavedModels inside the API process."""

    name = "local_tensorflow"
    ranker_names = RANKER_NAMES

    def __init__(self, user_tower: Any, rankers: dict[str, Any]) -> None:
        if set(rankers) != set(RANKER_NAMES):
            raise ValueError("both Wide & Deep and DCN rankers are required")
        self.user_tower = user_tower
        self.rankers = rankers

    @classmethod
    def load(cls, user_tower_path: Path, ranking_dir: Path):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        user_tower = tf.saved_model.load(str(user_tower_path)).signatures[
            "serving_default"
        ]
        rankers = {
            name: tf.saved_model.load(str(ranking_dir / name)).signatures[
                "serving_default"
            ]
            for name in RANKER_NAMES
        }
        return cls(user_tower, rankers)

    def user_embedding(self, user_id: int) -> np.ndarray:
        import tensorflow as tf

        output = self.user_tower(
            user_id=tf.constant([user_id], dtype=tf.int64)
        )["embedding"].numpy()
        return _validated_array(output, expected_rows=1, expected_columns=64)

    def ranker_logits(
        self, variant: str, features: dict[str, Any]
    ) -> np.ndarray:
        if variant not in self.rankers:
            raise ValueError(f"unknown ranker variant: {variant}")
        output = self.rankers[variant](**features)["logit"].numpy()
        row_count = len(next(iter(features.values())))
        return _validated_array(output, expected_rows=row_count)

    def health(self) -> dict:
        return {"backend": self.name, "ready": True, "models": sorted(RANKER_NAMES)}


class TritonHttpInference:
    """Use Triton's V2 JSON/HTTP inference protocol without a client SDK."""

    name = "triton_http"
    ranker_names = RANKER_NAMES

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(
            base_url=self.base_url, timeout=timeout_seconds
        )

    @staticmethod
    def _input(name: str, values: Any) -> dict:
        array = _to_numpy(values)
        if array.dtype.kind in {"i", "u"}:
            datatype = "INT64"
            array = array.astype(np.int64, copy=False)
        else:
            datatype = "FP32"
            array = array.astype(np.float32, copy=False)
        return {
            "name": name,
            "shape": list(array.shape),
            "datatype": datatype,
            "data": array.reshape(-1).tolist(),
        }

    def _infer(self, model: str, features: dict[str, Any], output: str) -> np.ndarray:
        response = self.client.post(
            f"/v2/models/{model}/infer",
            json={
                "inputs": [
                    self._input(name, value) for name, value in features.items()
                ],
                "outputs": [{"name": output}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        result = next(
            (item for item in payload.get("outputs", []) if item["name"] == output),
            None,
        )
        if result is None or "data" not in result:
            raise ValueError(f"Triton response did not contain output {output}")
        return np.asarray(result["data"], dtype=np.float32).reshape(result["shape"])

    def user_embedding(self, user_id: int) -> np.ndarray:
        features = {"user_id": np.asarray([user_id], dtype=np.int64)}
        output = self._infer("user_tower", features, "embedding")
        return _validated_array(output, expected_rows=1, expected_columns=64)

    def ranker_logits(
        self, variant: str, features: dict[str, Any]
    ) -> np.ndarray:
        if variant not in self.ranker_names:
            raise ValueError(f"unknown ranker variant: {variant}")
        output = self._infer(variant, features, "logit")
        row_count = len(next(iter(features.values())))
        return _validated_array(output, expected_rows=row_count)

    def health(self) -> dict:
        try:
            ready = self.client.get("/v2/health/ready")
            ready.raise_for_status()
            models = {}
            for name in ("user_tower", *sorted(self.ranker_names)):
                response = self.client.get(f"/v2/models/{name}/ready")
                models[name] = response.status_code == 200
            return {
                "backend": self.name,
                "ready": all(models.values()),
                "models": models,
            }
        except Exception as error:
            return {
                "backend": self.name,
                "ready": False,
                "error": type(error).__name__,
            }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validated_array(
    value: Any, *, expected_rows: int, expected_columns: int | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("inference returned non-finite values")
    expected_shape = (
        (expected_rows, expected_columns)
        if expected_columns is not None
        else (expected_rows,)
    )
    if array.shape != expected_shape:
        raise ValueError(
            f"inference returned shape {array.shape}; expected {expected_shape}"
        )
    return array
