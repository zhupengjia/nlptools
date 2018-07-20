#!/usr/bin/env python

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.multihead_attention import MultiheadAttention

from .decoder_base import Decoder_Base
from .transformer_encoder import LayerNorm, Linear, PositiuonalEmbedding

class TransformerDecoder(Decoder_Base):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

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
                encoder_out['encoder_padding_mask'],
                incremental_state,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

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

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


