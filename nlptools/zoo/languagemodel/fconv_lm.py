# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlptools.utils import eval_str_list
from ..modules.model_base import ModelBase

class FConvLanguageModel(ModelBase):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @classmethod
    def build_model(cls, vocab, tokens_per_sample, max_target_positions=None, decoder_embed_dim=128, decoder_layers=[(1268, 4)] * 13, decoder_attention=False, adaptive_softmax_cutoff=None, dropout=0.1, criterion=None, normalization_constant=0.5):
        """Build a new model instance."""

        if max_target_positions is not None:
            tokens_per_sample = max_target_positions

        decoder = FConvDecoder(
            vocab=vocab,
            embed_dim=decoder_embed_dim,
            out_embed_dim=decoder_embed_dim,
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
        return FConvLanguageModel(decoder)

    def forward(self, src_tokens):
        return self.decoder(src_tokens)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

