#!/usr/bin/env python

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertLayerNorm
from .multihead_attention import MultiheadAttention


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, bert_embedding, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.embedding = bert_embedding
        
        num_embeddings = self.embedding.word_embeddings.num_embeddings
        embedding_dim = self.embedding.word_embeddings.embedding_dim

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embedding_dim, num_attention_heads, intermediate_size, dropout)
            for i in range(num_hidden_layers)
        ])
        
        self.fc3 = nn.Linear(embedding_dim, num_embeddings)
        self.fc3.weight = self.embedding.word_embeddings.weight

        self.layer_norm = BertLayerNorm(embedding_dim, eps=1e-12)

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask):
        # embed tokens and positions
        x = self.embedding(prev_output_tokens)
        

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        encoder_padding_mask = encoder_padding_mask.byte()
        encoder_padding_mask = ~encoder_padding_mask # for mask fill

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                encoder_padding_mask
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc3(x)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embedding_dim, num_attention_heads, intermediate_size, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_attn = MultiheadAttention(
            embedding_dim, num_attention_heads,
            dropout=self.dropout,
        )

        self.encoder_attn = MultiheadAttention(
            embedding_dim, num_attention_heads,
            dropout=self.dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, embedding_dim)
        self.layer_norms = nn.ModuleList([BertLayerNorm(embedding_dim, eps=1e-12) for i in range(3)])

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
            key_padding_mask=encoder_padding_mask
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

