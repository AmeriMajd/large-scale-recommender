"""Wide & Deep and Deep & Cross binary relevance rankers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import tensorflow_recommenders as tfrs
import tf_keras as keras

from recommender.ranking.data import MODEL_FEATURES, NUMERIC_FEATURES


class FeatureEncoder(keras.layers.Layer):
    """Combine learned ID embeddings with scaled dense features."""

    def __init__(
        self,
        user_ids: np.ndarray,
        movie_ids: np.ndarray,
        embedding_dimension: int,
        seed: int,
    ) -> None:
        super().__init__(name="feature_encoder")
        self.user_lookup = keras.layers.IntegerLookup(
            vocabulary=user_ids.tolist(), mask_token=None, num_oov_indices=1
        )
        self.movie_lookup = keras.layers.IntegerLookup(
            vocabulary=movie_ids.tolist(), mask_token=None, num_oov_indices=1
        )
        self.user_embedding = keras.layers.Embedding(
            self.user_lookup.vocabulary_size(),
            embedding_dimension,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed),
        )
        self.movie_embedding = keras.layers.Embedding(
            self.movie_lookup.vocabulary_size(),
            embedding_dimension,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
        )

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        numeric = tf.stack(
            [tf.cast(features[name], tf.float32) for name in NUMERIC_FEATURES],
            axis=1,
        )
        return tf.concat(
            [
                self.user_embedding(self.user_lookup(features["user_id"])),
                self.movie_embedding(self.movie_lookup(features["movie_id"])),
                numeric,
            ],
            axis=1,
        )


class WideAndDeepRanker(keras.Model):
    """Parallel linear and MLP relevance model."""

    def __init__(self, encoder: FeatureEncoder, deep_layer_sizes: tuple[int, ...]):
        super().__init__(name="wide_and_deep_ranker")
        self.encoder = encoder
        self.wide = keras.layers.Dense(1, name="wide_logit")
        self.deep = keras.Sequential(
            [keras.layers.Dense(size, activation="relu") for size in deep_layer_sizes]
            + [keras.layers.Dense(1)],
            name="deep_tower",
        )

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        encoded = self.encoder(features)
        return tf.squeeze(self.wide(encoded) + self.deep(encoded), axis=1)


class DcnRanker(keras.Model):
    """Parallel explicit cross network and implicit deep network."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        deep_layer_sizes: tuple[int, ...],
        cross_layers: int,
    ) -> None:
        super().__init__(name="dcn_ranker")
        self.encoder = encoder
        self.cross_network = [
            tfrs.layers.dcn.Cross(projection_dim=32, name=f"cross_{index}")
            for index in range(cross_layers)
        ]
        self.deep = keras.Sequential(
            [keras.layers.Dense(size, activation="relu") for size in deep_layer_sizes],
            name="deep_tower",
        )
        self.logit = keras.layers.Dense(1)

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        encoded = self.encoder(features)
        crossed = encoded
        for layer in self.cross_network:
            crossed = layer(encoded, crossed)
        deep = self.deep(encoded)
        return tf.squeeze(self.logit(tf.concat([crossed, deep], axis=1)), axis=1)


def build_ranker(
    kind: str,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    embedding_dimension: int,
    deep_layer_sizes: tuple[int, ...],
    cross_layers: int,
    learning_rate: float,
    seed: int,
) -> keras.Model:
    """Create one deterministically initialized binary relevance ranker."""
    keras.utils.set_random_seed(seed)
    encoder = FeatureEncoder(user_ids, movie_ids, embedding_dimension, seed)
    if kind == "wide_and_deep":
        model = WideAndDeepRanker(encoder, deep_layer_sizes)
    elif kind == "dcn":
        model = DcnRanker(encoder, deep_layer_sizes, cross_layers)
    else:
        raise ValueError(f"unknown ranking model kind: {kind}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.AUC(name="auc", from_logits=True),
            keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.0),
        ],
    )
    return model


def export_ranker(model: keras.Model, path: Path) -> None:
    """Export a model with one stable dictionary-based serving signature."""
    signature = {
        name: tf.TensorSpec(
            [None],
            tf.int64 if name in {"user_id", "movie_id"} else tf.float32,
            name=name,
        )
        for name in MODEL_FEATURES
    }

    @tf.function(input_signature=[signature])
    def serve(features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        return {"logit": model(features, training=False)}

    tf.saved_model.save(model, str(path), signatures={"serving_default": serve})
