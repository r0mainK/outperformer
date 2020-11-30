import torch
from torch.nn import Dropout
from torch.nn import Embedding
from torch.nn import GELU
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module

from .fast_attention import FastSelfAttention
from .reversible import ReversibleStack


class EmbeddingLayer(Module):
    def __init__(self, vocab_size, max_seq_length, d_model):
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(max_seq_length, d_model)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, x):
        seq_length = x.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embeddings(x) + self.position_embeddings(position_ids)


class ChunkedFeedForwardLayer(Module):
    def __init__(self, d_model, d_ff, dropout_rate, c):
        super(ChunkedFeedForwardLayer, self).__init__()
        self.linear_1 = Linear(d_model, d_ff)
        self.linear_2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout_rate)
        self.gelu = GELU()
        self.c = c

    def forward(self, x):
        chunks = x.chunk(self.c, dim=1)
        chunks = [self.linear_2(self.dropout(self.gelu(self.linear_1(chunk)))) for chunk in chunks]
        return torch.cat(chunks, dim=1)


class Sublayer(Module):
    def __init__(self, d_model, dropout_rate, sublayer):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout_rate)
        self.sublayer = sublayer

    def forward(self, x):
        return self.norm(self.dropout(self.sublayer(x)))


class OutPerformer(Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        d_model,
        d_ff,
        n_layers,
        h,
        m,
        dropout_rate,
        use_hyperbolic,
    ):
        super(OutPerformer, self).__init__()
        self.embedding_layer = Sublayer(
            d_model, dropout_rate, EmbeddingLayer(vocab_size, max_seq_length, d_model)
        )

        stack = [
            (
                Sublayer(d_model, dropout_rate, FastSelfAttention(d_model, h, m, use_hyperbolic)),
                Sublayer(
                    d_model, dropout_rate, ChunkedFeedForwardLayer(d_model, d_ff, dropout_rate)
                ),
            )
            for _ in range(n_layers)
        ]
        self.reversible_stack = ReversibleStack(stack)
        self.final_layer = Sublayer(d_model, dropout_rate, Linear(d_model, d_model))

    def forward(self, x):
        return self.final_layer(self.reversible_stack(self.embedding_layer(x)))

    def redraw_orf(self):
        for reversible_layer in self.reversible_stack.layers:
            reversible_layer.layer_1.layer.sublayer.redraw_orf()

    def update_chunk_ratio(self, c):
        for reversible_layer in self.reversible_stack.layers:
            reversible_layer.layer_2.layer.sublayer.c = c
