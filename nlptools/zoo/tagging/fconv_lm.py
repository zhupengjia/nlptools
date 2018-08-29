#!/usr/bin/env python

import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from nlptools.utils import eval_str_list
from .lm_base import LanguageModelBase
from ..encoders.fconv_decoder import FConvDecoder

class FConvLanguageModel(LanguageModelBase):
    def __init__(self, vocab, pretrained_embed, tokens_per_sample=1024, max_target_positions=None, decoder_layers=[(1268, 4)] * 13, decoder_attention=False, adaptive_softmax_cutoff=None, dropout=0.1, criterion=None, normalization_constant=0.5, device='cpu'):
        super().__init__(vocab, pretrained_embed, device)
        if max_target_positions is not None:
            tokens_per_sample = max_target_positions
        
        self.decoder = FConvDecoder(
            vocab=vocab,
            pretrained_embed = self.pretrained_embed
            out_embed_dim=vocab.embedding_dim,
            max_positions=tokens_per_sample,
            convolutions=decoder_layers,
            attention=decoder_attention,
            dropout=dropout,
            share_embed=False,
            positional_embeddings=False,
            adaptive_softmax_cutoff=(
                eval_str_list(adaptive_softmax_cutoff, type=int)
                if criterion == 'adaptive_loss' else None
            ),
            normalization_constant=normalization_constant,
        )


    def forward(self, src_tokens):
        return self.decoder(src_tokens)

    def max_positions(self):
        return self.decoder.max_positions()

    


