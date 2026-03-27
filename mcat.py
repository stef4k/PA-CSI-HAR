"""
mcat.py – Multi-scale Convolution-Augmented Transformer (MCAT) layer.

Bug-fixes vs original:
- Encoder used `self.layers = []` which shadows the Keras built-in `layers`
  property and breaks sub-layer weight tracking.  Renamed to `self._enc_layers`.
- HAR_CNN used `self.encoders = []` with subsequent appends; replaced with
  list-comprehension so Keras sees all layers at attribute-assignment time.
- MultiHeadAttention stored Dense layers in `self.linears`; same fix applied.
- Hardcoded division-by-3 in HAR_CNN replaced with len(filter_sizes).
"""

import math
import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LayerNorm(layers.Layer):
    def __init__(self, features, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.a_2 = float(1)
        self.b_2 = float(0)
        self.features = features
        self.eps = eps

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(layers.Layer):
    """Residual connection + layer norm (pre-norm variant)."""

    def __init__(self, size, dropout, **kwargs):
        super().__init__(**kwargs)
        self.norm    = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def scaled_dot_attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention (TF implementation)."""
    d_k    = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)
    if mask is not None:
        scores += mask * -1e9
    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return tf.matmul(p_attn, value), p_attn


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(layers.Layer):
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h   = h
        # Four projections: Q, K, V, output  – stored as a list attribute
        self.linears = [layers.Dense(d_model) for _ in range(4)]
        self.attn    = None
        self.dropout = layers.Dropout(dropout)

    def call(self, query, key, value, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, 1)
        nbatches = tf.shape(query)[0]

        # Project and split into h heads
        query, key, value = [
            tf.transpose(
                tf.reshape(lin(x), (nbatches, -1, self.h, self.d_k)),
                perm=[0, 2, 1, 3],
            )
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]

        x, self.attn = scaled_dot_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = tf.reshape(
            tf.transpose(x, perm=[0, 2, 1, 3]),
            (nbatches, -1, self.h * self.d_k),
        )
        return self.linears[3](x)


# ---------------------------------------------------------------------------
# Multi-scale CNN (feed-forward replacement inside MCAT)
# ---------------------------------------------------------------------------

class HAR_CNN(layers.Layer):
    """Multi-scale 1-D CNN used as the feed-forward block inside MCAT."""

    def __init__(self, d_model, d_ff, filters, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filter_sizes = list(filters)
        self.dropout = layers.Dropout(dropout)
        self.bn      = layers.BatchNormalization(axis=1)
        self.relu    = layers.Activation("relu")
        # Build all Conv1D layers at init time so Keras tracks their weights
        self.encoders = [
            layers.Conv1D(
                filters=int(d_ff),
                kernel_size=fs,
                data_format="channels_first",
                padding="same",
            )
            for fs in self.filter_sizes
        ]

    def call(self, data, training=False):
        data = tf.cast(data, tf.float32)
        data_t = tf.transpose(data, perm=[0, 2, 1])
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(data_t)
            f_map = self.relu(self.dropout(self.bn(f_map, training=training),
                                           training=training))
            enc_outs.append(tf.expand_dims(f_map, axis=1))

        # Average across filter branches
        stacked = tf.concat(enc_outs, axis=1)           # (B, n_filters, d_ff, L)
        averaged = tf.divide(
            tf.reduce_sum(stacked, axis=1),
            float(len(self.filter_sizes)),
        )
        return tf.transpose(averaged, perm=[0, 2, 1])


# ---------------------------------------------------------------------------
# Encoder layer & stacked encoder
# ---------------------------------------------------------------------------

class EncoderLayer(layers.Layer):
    """Single MCAT encoder layer: multi-head attention + HAR_CNN."""

    def __init__(self, size, self_attn, feed_forward, dropout, **kwargs):
        super().__init__(**kwargs)
        self.self_attn    = self_attn
        self.feed_forward = feed_forward
        self.sublayer_0   = SublayerConnection(size, dropout)
        self.sublayer_1   = SublayerConnection(size, dropout)
        self.size         = size

    def call(self, x):
        x = self.sublayer_0(x, lambda z: self.self_attn(z, z, z))
        return self.sublayer_1(x, self.feed_forward)


class Encoder(layers.Layer):
    """Stack of N EncoderLayer instances (shared weights – same layer × N).

    Note: the original code reused one EncoderLayer instance N times, which
    implements weight-tied depth-N stacking. We preserve that behaviour while
    fixing the `self.layers` name collision with the Keras built-in property
    by storing the stack as `self._enc_layers`.
    """

    def __init__(self, layer, N, **kwargs):
        super().__init__(**kwargs)
        # Same instance N times → weight tying across depth (matches paper code)
        self._enc_layers = [layer] * N
        self.norm = LayerNorm(layer.size)

    def call(self, x, mask=None):
        for enc_layer in self._enc_layers:
            x = enc_layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# MCAT
# ---------------------------------------------------------------------------

class MCAT(layers.Layer):
    """Multi-scale Convolution Augmented Transformer.

    Parameters
    ----------
    hidden_dim : int
        Feature dimensionality (d_model).
    N : int
        Number of encoder layers (weight-tied).
    H : int
        Number of attention heads.
    total_size : int
        Kept for API compatibility (not used internally).
    filters : list[int]
        Kernel sizes for the internal multi-scale CNN.
    """

    def __init__(
        self,
        hidden_dim,
        N,
        H,
        total_size,
        filters=(1, 3, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = Encoder(
            EncoderLayer(
                hidden_dim,
                MultiHeadAttention(H, hidden_dim),
                HAR_CNN(hidden_dim, hidden_dim, filters),
                0.1,
            ),
            N,
        )

    def call(self, x):
        return self.model(x)
