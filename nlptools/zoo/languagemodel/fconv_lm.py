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

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)

from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel,
    FairseqLanguageModel, register_model, register_model_architecture,
)


class FConvLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if hasattr(args, 'max_target_positions'):
            args.tokens_per_sample = args.max_target_positions

        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.tokens_per_sample,
            share_embed=False,
            positional_embeddings=False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            normalization_constant=args.normalization_constant,
        )
        return FConvLanguageModel(decoder)

    def forward(self, src_tokens):
        return self.decoder(src_tokens)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

