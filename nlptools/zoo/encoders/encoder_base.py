#!/usr/bin/env python

import torch.nn as nn


class Encoder_Base(nn.Module):
    """Base class for encoders."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, src_tokens, src_lengths):
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        raise NotImplementedError

    def upgrade_state_dict(self, state_dict):
        return state_dict
