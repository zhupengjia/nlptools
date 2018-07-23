#!/usr/bin/env python

import torch,  sys
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.fconv_encoder import FConvEncoder
from ..encoders.fconv_decoder import FConvDecoder
from .seq2seq_base import Seq2SeqBase


class FConvSeq2Seq(Seq2SeqBase):
    def __init__(self, encoder_vocab, decoder_vocab=None, pretrained_embed=True, encoder_layers=((256, 3),)*4, decoder_layers=((256, 3),)*3, decoder_attention=True,  dropout=0.1, max_source_positions=1024, max_target_positions=1024, share_embed=False, decoder_share_embed=False, normalization_constant=0.5, device='cpu'):
        super().__init__(encoder_vocab, decoder_vocab, pretrained_embed, share_embed, decoder_share_embed, device)
        
        self.encoder = FConvEncoder(
            vocab = self.encoder_vocab,
            pretrained_embed = pretrained_embed,
            convolutions=encoder_layers,
            dropout=dropout,
            max_positions=max_source_positions,
            normalization_constant=normalization_constant
        )
        
        self.decoder = FConvDecoder(
            vocab = self.decoder_vocab,
            out_embed_dim=self.decoder_vocab.embedding_dim,
            pretrained_embed = pretrained_embed,
            convolutions=decoder_layers,
            attention=decoder_attention,
            dropout=dropout,
            max_positions=max_target_positions,
            share_embed=decoder_share_embed,
            normalization_constant=normalization_constant
        )

        if share_embed:
            self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight
        
        self.encoder.num_attention_layers = sum(layer is not None for layer in self.decoder.attention)

    
