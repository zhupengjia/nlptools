#!/usr/bin/env python

import torch, math
import torch.nn as nn
import torch.nn.functional as F

from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

from .encoder_base import Encoder_Base


class TransformerEncoder(Encoder_Base):
    """Transformer encoder."""

    def __init__(self, vocab, pretrained_embed=True, layers=3, attention_heads=4, ffn_embed_dim=512, dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, left_pad=True):
        super().__init__(vocab)
        self.dropout = dropout

        num_embeddings = vocab.vocab_size
        embed_dim = vocab.embedding_dim
        self.padding_idx = vocab.PAD_ID

        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if pretrained_embed:
            self.embed_tokens.weight.data = torch.FloatTensor(vocab.dense_vectors())

        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
                    embed_dim,
                    self.padding_idx,
                    left_pad,
                    1024
                )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(embed_dim, attention_heads, ffn_embed_dim, dropout, attention_dropout, relu_dropout)
            for i in range(layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class TransformerEncoderLayer(nn.Module):
    """
        Encoder layer block.
    """

    def __init__(self, embed_dim, attention_heads, ffn_embed_dim, dropout, attention_dropout, relu_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, attention_heads,
            dropout=attention_dropout,
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.fc1 = Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.layer_norms[0](x)        

        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


