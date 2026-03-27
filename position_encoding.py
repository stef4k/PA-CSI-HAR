"""
position_encoding.py – TensorFlow-native positional encodings.

PE  : standard sinusoidal encoding (Vaswani et al., 2017)
GRE : Gaussian Range Encoding (Li et al., 2021 – THAT model),
      re-implemented natively in TF so parameters are trained end-to-end.

Bug-fix vs original: the original GRE subclassed nn.Module (PyTorch) and
returned `.detach().numpy()`, meaning the Gaussian means/sigma/embedding
were never updated during TF training. This version is a Keras Layer with
proper add_weight() so those parameters receive gradients.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class PE(layers.Layer):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_seq_length=500, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_layer = layers.Dropout(dropout)

        pe = np.zeros((max_seq_length, d_model), dtype=np.float32)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * i + 1) / d_model))
                    )
        self.pe = tf.constant(pe.reshape(1, max_seq_length, d_model))

    def call(self, x, training=False):
        x = x * math.sqrt(self.d_model)
        seq_length = tf.shape(x)[1]
        x = x + self.pe[:, :seq_length]
        return self.dropout_layer(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


class GRE(layers.Layer):
    """Gaussian Range Encoding – TF-native re-implementation.

    Each position is encoded as a weighted sum of K learnable embeddings,
    where the weights are given by softmax over K Gaussian basis functions.

    Matches the original formulation from Li et al. (2021):
        log_p = -a² / (2·σ) - log(σ) / 2,   a = position - μ
        M = softmax(log_p, axis=K)
        pos_enc = M @ embedding              shape: (total_size, d_model)
    """

    def __init__(self, d_model: int, total_size: int, K: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.total_size = total_size
        self.K = K

        # Fixed positions matrix: shape (total_size, K),
        # each row is [pos, pos, ..., pos] replicated K times.
        positions = np.tile(
            np.arange(total_size, dtype=np.float32).reshape(-1, 1),
            (1, K),
        )
        self.positions = tf.constant(positions)  # not a learnable weight

        # Evenly spaced initial Gaussian means
        interval = total_size / K
        self._mu_init = np.array(
            [i * interval for i in range(K)], dtype=np.float32
        ).reshape(1, K)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="gre_embedding",
            shape=(self.K, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.mu = self.add_weight(
            name="gre_mu",
            shape=(1, self.K),
            initializer=tf.keras.initializers.Constant(self._mu_init),
            trainable=True,
        )
        self.sigma = self.add_weight(
            name="gre_sigma",
            shape=(1, self.K),
            initializer=tf.keras.initializers.Constant(
                np.full((1, self.K), 50.0, dtype=np.float32)
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # a: (total_size, K)
        a = self.positions - self.mu
        # Unnormalised log-Gaussian (matches original PyTorch formulation)
        log_p = (
            -tf.square(a) / (2.0 * self.sigma + 1e-8)
            - tf.math.log(tf.abs(self.sigma) + 1e-8) / 2.0
        )
        # Softmax over K: each position gets a convex combination of embeddings
        M = tf.nn.softmax(log_p, axis=1)          # (total_size, K)
        pos_enc = tf.matmul(M, self.embedding)    # (total_size, d_model)
        pos_enc = tf.expand_dims(pos_enc, 0)      # (1, total_size, d_model)
        return x + pos_enc

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"d_model": self.d_model, "total_size": self.total_size, "K": self.K}
        )
        return cfg
