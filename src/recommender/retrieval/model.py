"""TensorFlow Recommenders two-tower retrieval model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
import tensorflow_recommenders as tfrs  # noqa: E402
import tf_keras as keras  # noqa: E402


class IdEmbeddingTower(keras.Model):
    """Map known integer IDs into normalized vectors with one OOV bucket."""

    def __init__(
        self,
        vocabulary: np.ndarray | list[int],
        embedding_dimension: int,
        seed: int,
        name: str,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._vocabulary = np.asarray(vocabulary, dtype=np.int64)
        self._embedding_dimension = embedding_dimension
        self._seed = seed
        self.lookup = keras.layers.IntegerLookup(
            vocabulary=self._vocabulary.tolist(),
            mask_token=None,
            num_oov_indices=1,
            name=f"{name}_lookup",
        )
        self.embedding = keras.layers.Embedding(
            input_dim=self.lookup.vocabulary_size(),
            output_dim=embedding_dimension,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed),
            name=f"{name}_embedding",
        )

    def call(self, ids: tf.Tensor) -> tf.Tensor:
        embeddings = self.embedding(self.lookup(ids))
        return tf.math.l2_normalize(embeddings, axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "vocabulary": self._vocabulary.tolist(),
                "embedding_dimension": self._embedding_dimension,
                "seed": self._seed,
            }
        )
        return config


class TwoTowerRetrievalModel(tfrs.models.Model):
    """Train user and movie towers using sampled-softmax retrieval loss."""

    def __init__(self, user_tower: IdEmbeddingTower, item_tower: IdEmbeddingTower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=None,
            remove_accidental_hits=True,
        )

    def compute_loss(
        self,
        features: dict[str, tf.Tensor] | tuple[dict[str, tf.Tensor]],
        training=False,
    ) -> tf.Tensor:
        del training
        if isinstance(features, tuple):
            features = features[0]
        user_embeddings = self.user_tower(features["user_id"])
        item_embeddings = self.item_tower(features["movie_id"])
        return self.retrieval_task(
            user_embeddings,
            item_embeddings,
            candidate_ids=features["movie_id"],
            compute_metrics=False,
            compute_batch_metrics=False,
        )


def build_model(
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    embedding_dimension: int,
    learning_rate: float,
    seed: int,
) -> TwoTowerRetrievalModel:
    """Create and compile a deterministic ID-only development model."""
    keras.utils.set_random_seed(seed)
    user_tower = IdEmbeddingTower(
        user_ids, embedding_dimension, seed, name="user_tower"
    )
    item_tower = IdEmbeddingTower(
        movie_ids, embedding_dimension, seed + 1, name="item_tower"
    )
    model = TwoTowerRetrievalModel(user_tower, item_tower)
    model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate))
    return model


def embed_ids(
    tower: IdEmbeddingTower,
    ids: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Embed IDs in bounded batches and return contiguous float32 vectors."""
    dataset = tf.data.Dataset.from_tensor_slices(ids).batch(batch_size)
    parts = [tower(batch, training=False).numpy() for batch in dataset]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def export_tower(tower: IdEmbeddingTower, path: Path, input_name: str) -> None:
    """Export one tower with a stable named serving signature."""

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64, name=input_name)]
    )
    def serve(ids: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"embedding": tower(ids, training=False)}

    tf.saved_model.save(tower, str(path), signatures={"serving_default": serve})
