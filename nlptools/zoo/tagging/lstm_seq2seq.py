#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq2seq_base import Seq2SeqBase
from ..encoders.lstm_encoder import LSTMEncoder 
from ..encoders.lstm_decoder import LSTMDecoder


def LSTMSeq2Seq(Seq2SeqBase):
    def __init__(self, encoder_vocab, decoder_vocab=None, pretrained_embed=True, encoder_hidden_size=512, decoder_hidden_size=512, encoder_layers=1, decoder_layers=1, bidirectional=False, attention=True, share_embed=False, decoder_share_embed=False,  dropout_in=0.1, dropout_out=0.1, device='cpu'):

        super().__init__(encoder_vocab, decoder_vocab, pretrained_embed, share_embed, decoder_share_embed, device)

        self.encoder = LSTMEncoder(
            vocab = self.encoder_vocab,
            pretrained_embed = pretrained_embed, 
            hidden_size = encoder_hidden_size,
            num_layers = encoder_layers,
            bidirectional = bidirectional,
            dropout_in = dropout_in,
            dropout_out = dropout_out
        )
        self.decoder = LSTMDecoder(
            vocab = self.decoder_vocab,
            out_embed_dim=self.decoder_vocab.embedding_dim,
            pretrained_embed = pretrained_embed, 
            hidden_size = decoder_hidden_size,
            num_layers = decoder_layers,
            attention = attention,
            encoder_output_units = self.encoder.output_units,
            dropout_in = dropout_in,
            dropout_out = dropout_out
        )

        if share_embed:
            self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight

