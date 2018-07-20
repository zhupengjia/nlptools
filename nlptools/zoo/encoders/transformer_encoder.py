#!/usr/bin/env python

import torch, math
import torch.nn as nn
import torch.nn.functional as F

from ..modules.learned_positional_embedding import LearnedPositionalEmbedding
from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

from .encoder_base import Encoder_Base


class TransformerEncoder(Encoder_Base):
    """Transformer encoder."""

    def __init__(self, vocab, predtrained_embed=True, encoder_layers=3, encoder_learned_pos=False, encoder_attention_heads=4, encoder_ffn_embed_dim=512, encoder_normalize_before=False, dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, left_pad=True):
        super().__init__(vocab)
        self.dropout = dropout

        num_embeddings = vocab.vocab_size
        embed_dim = vocab.embedding_dim
        self.padding_idx = vocab.PAD_ID

        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if pretrained_embed:
            self.embed_tokens.weight.data = torch.FloatTensor(vocab.dense_vectors())

        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(embed_dim, encoder_attention_heads, encoder_normalize_before, encoder_ffn_embed_dim, dropout, attention_dropout, relu_dropout)
            for i in range(encoder_layers)
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

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out'] is not None:
            encoder_out_dict['encoder_out'] = \
                encoder_out_dict['encoder_out'].index_select(1, new_order)
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, encoder_embed_dim, encoder_attention_heads, encoder_normalize_before, encoder_ffn_embed_dim, dropout, attention_dropout, relu_dropout):
        super().__init__()
        self.embed_dim = encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, encoder_attention_heads,
            dropout=attention_dropout,
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.normalize_before = encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear(encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
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


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m


