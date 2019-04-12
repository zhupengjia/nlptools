#!/usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from fairseq.models.transformer import TransformerDecoderLayer
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class TransformerDecoder(FairseqIncrementalDecoder):
    """
        Transformer decoder. Modified from Fairseq
        Worked with pretrained BERT model from pytorch_pretrained_bert
    """

    def __init__(self, bert_embedding, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024, dropout=0.1, shared_embed=True):
        super().__init__(dictionary=None)
        self.dropout = dropout
       
        self.word_embedding = bert_embedding.word_embeddings
        self.position_embedding = bert_embedding.position_embeddings
        self.layer_norm = bert_embedding.LayerNorm

        num_embeddings = self.word_embedding.num_embeddings
        embedding_dim = self.word_embedding.embedding_dim

        args = SimpleNamespace(decoder_embed_dim = embedding_dim,
                               decoder_attention_heads = num_attention_heads,
                               dropout = dropout,
                               attention_dropout = dropout,
                               relu_dropout = dropout,
                               decoder_normalize_before = False,
                               decoder_ffn_embed_dim = intermediate_size
                              )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(num_hidden_layers)
        ])
        
        self.fc3 = nn.Linear(embedding_dim, num_embeddings, bias=False)
        if shared_embed:
            self.fc3.weight = self.word_embedding.weight

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask, time_step=0, incremental_state=None):
        # embed tokens and positions
        word_embeddings = self.word_embedding(prev_output_tokens)
        position_ids = torch.arange(time_step, prev_output_tokens.size(1), dtype=torch.long,
                                   device=prev_output_tokens.device)
        position_embeddings = self.position_embedding(position_ids)
        x = word_embeddings + position_embeddings
        x = self.layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        encoder_padding_mask = encoder_padding_mask.byte()
        encoder_padding_mask = ~encoder_padding_mask # for mask fill

        # decoder layers
        inner_states = [x]
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                encoder_padding_mask,
                incremental_state
            )
            inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc3(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        return  self.position_embedding.num_embeddings

