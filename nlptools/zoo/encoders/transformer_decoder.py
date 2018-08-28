#!/usr/bin/env python

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

from .decoder_base import Decoder_Base
from .transformer_encoder import LayerNorm, Embedding, Linear

class TransformerDecoder(Decoder_Base):
    """Transformer decoder."""

    def __init__(self, vocab, pretrained_embed=True, layers=6, attention_heads=8, ffn_embed_dim=1024, share_embed=True, dropout=0.1, left_pad=False):
        super().__init__(vocab)
        self.dropout = dropout

        num_embeddings = vocab.vocab_size
        embed_dim = vocab.embedding_dim
        padding_idx = vocab.PAD_ID

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if pretrained_embed:
            self.embed_tokens.weight.data = torch.FloatTensor(vocab.dense_vectors())

        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, 
            padding_idx,
            left_pad=left_pad,
            1024
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_dim, attention_heads, ffn_embed_dim, dropout)
            for i in range(layers)
        ])
        
        self.fc3 = Linear(embed_dim, num_embeddings)
        if share_embed:
            self.fc3.weight = self.embed_tokens.weight


    def forward(self, prev_output_tokens, encoder_out):
        # embed positions
        positions = self.embed_positions(prev_output_tokens)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask']
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc3(x)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embed_dim, attention_heads, ffn_embed_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.self_attn = MultiheadAttention(
            self.embed_dim, attention_heads,
            dropout=self.dropout,
        )
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, attention_heads,
            dropout=self.dropout,
        )
        self.fc1 = Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask):
        residual = x
        x = self.layer_norms[0](x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[2](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x, attn


