#!/usr/bin/env python

import torch.nn as nn

class ModelBase(nn.Module):

    def __init__(self):
        super().__init__()

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()




