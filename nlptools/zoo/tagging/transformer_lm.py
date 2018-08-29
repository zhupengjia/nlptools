#!/usr/bin/env python

import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from nlptools.utils import eval_str_list
from .lm_base import LanguageModelBase
from ..encoders.transformer_decoder import TransformerDecoder

class TransformerLanguageModel(LanguageModelBase):
    def __init__(self, vocab, pretrained_embed=True, layers=6, attention_heads=8, ffn_embed_dim=512 shared_embed=False, device='cpu'):
        super().__init__(vocab, pretrained_embed, device)

        sefl.decoder = TransformerDecoder(
                    vocab = vocab,
                    pretrained_embed = self.pretrained_embed,
                    layers = layers,
                    attention_heads = attention_heads,
                    ffn_embed_dim = ffn_embed_dim,
                    share_embed = share_embed,
                    dropout = dropout
                )

    def forward(self, src_tokens):
        return self.decoder(src_tokens)

    def max_positions(self):
        return self.decoder.max_positions()
