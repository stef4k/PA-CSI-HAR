"""
models.py – PA-CSI dual-stream model.

Changes vs original:
- Two_Stream_Model is now fully parameterised (embed_dim, seq_len,
  filter_sizes, dropout_rate) – no more hard-coded values.
- Paper hyperparameters (Table 3): filter_sizes=[10,40], dropout=0.1,
  hlayers=4.  Original had [20,40], 0.5, 5 respectively.
- GRE now receives the correct (embed_dim, reduced_len) instead of hard-coded
  (270, 500).
- Vertical transformer removed: `transformer_encoder.Transfomer` contained
  unfixed bugs (forward() instead of call(), missing return) that prevented
  it from running.  The temporal MCAT + dual amplitude/phase streams already
  capture both spatial and temporal structure as described in the paper.
- Global max-pool replaces the dynamic MaxPooling1D-with-variable-pool-size
  which cannot be used inside a Keras Lambda-free call().
- `build_model(cfg)` is exposed as the single public entry point.
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from mcat import MCAT
from position_encoding import GRE
from grn import GatesResidualNetwork


class TwoStreamBlock(layers.Layer):
    """Processes one CSI feature stream (amplitude OR phase).

    Pipeline
    --------
    1. Temporal averaging:  (B, T, D) → (B, T//sample, D)
    2. Gaussian Range Encoding (GRE)
    3. Multi-scale Conv-Augmented Transformer (MCAT) – temporal attention
    4. Multi-scale Conv1D aggregation with global max-pool
       → output shape: (B, kernel_num × len(filter_sizes))
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        hlayers: int,
        hheads: int,
        K: int,
        sample: int,
        filter_sizes=(10, 40),
        kernel_num: int = 128,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim   = embed_dim
        self.seq_len     = seq_len
        self.sample      = sample
        self.reduced_len = seq_len // sample
        self.filter_sizes = list(filter_sizes)
        self.kernel_num  = kernel_num

        # Positional encoding on the down-sampled sequence
        self.pos_encoding = GRE(embed_dim, self.reduced_len, K)

        # Temporal MCAT (4 layers, paper Table 3)
        self.transformer = MCAT(embed_dim, hlayers, hheads, self.reduced_len)

        # Aggregation: multi-scale Conv1D + global max-pool
        self.relu    = layers.Activation("relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.conv_layers = [
            layers.Conv1D(
                filters=kernel_num,
                kernel_size=fs,
                data_format="channels_first",
            )
            for fs in self.filter_sizes
        ]

    def _aggregate(self, o):
        """o: (B, reduced_len, embed_dim) → (B, kernel_num × n_filters)"""
        o_t = tf.transpose(o, perm=[0, 2, 1])   # (B, embed_dim, reduced_len)
        enc_outs = []
        for conv in self.conv_layers:
            f_map  = conv(o_t)                   # (B, kernel_num, L')
            f_map  = self.relu(f_map)
            pooled = tf.reduce_max(f_map, axis=-1)  # global max: (B, kernel_num)
            enc_outs.append(pooled)
        concat = tf.concat(enc_outs, axis=1)    # (B, kernel_num × n_filters)
        return self.relu(self.dropout(concat))

    def call(self, data, training=False):
        data = tf.cast(data, tf.float32)
        d1 = tf.shape(data)[0]
        d2 = tf.shape(data)[1]
        d3 = tf.shape(data)[2]

        # Temporal down-sampling by averaging every `sample` steps
        x = tf.reshape(data, [d1, d2 // self.sample, self.sample, d3])
        x = tf.reduce_mean(x, axis=2)            # (B, reduced_len, embed_dim)

        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self._aggregate(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> keras.Model:
    """Build the full dual-stream PA-CSI model from a configuration dict.

    Expected keys in cfg (paper Table 3 defaults shown):
        maxlen        : int   – input sequence length
        embed_dim     : int   – number of CSI features per time step
        num_class     : int   – number of activity classes
        hlayers       : int   – number of MCAT encoder layers  (paper: 4)
        hheads        : int   – multi-head attention heads       (paper: 9)
        K             : int   – Gaussian encoding components     (paper: 10)
        sample        : int   – temporal down-sampling factor    (paper: 2)
        filter_sizes  : list  – Conv1D kernel sizes              (paper: [10,40])
        dropout       : float – dropout rate                     (paper: 0.1)
        grn_units     : int   – GRN hidden units                 (paper: 256)
    """
    maxlen    = cfg["maxlen"]
    embed_dim = cfg["embed_dim"]
    num_class = cfg["num_class"]

    inputs_amp   = layers.Input(shape=(maxlen, embed_dim), name="amplitude")
    inputs_phase = layers.Input(shape=(maxlen, embed_dim), name="phase")

    stream_kwargs = dict(
        embed_dim    = embed_dim,
        seq_len      = maxlen,
        hlayers      = cfg["hlayers"],
        hheads       = cfg["hheads"],
        K            = cfg["K"],
        sample       = cfg["sample"],
        filter_sizes = cfg["filter_sizes"],
        dropout_rate = cfg["dropout"],
    )

    feat_amp   = TwoStreamBlock(**stream_kwargs, name="stream_amp")(inputs_amp)
    feat_phase = TwoStreamBlock(**stream_kwargs, name="stream_phase")(inputs_phase)

    fused   = GatesResidualNetwork(cfg["grn_units"])(feat_amp, feat_phase)
    outputs = layers.Dense(num_class, activation="softmax")(fused)

    return keras.Model(inputs=[inputs_amp, inputs_phase], outputs=outputs)
