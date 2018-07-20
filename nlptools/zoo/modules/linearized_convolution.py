#!/usr/bin/env python

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .conv_tbc import ConvTBC


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input, incremental_state=None):
        """
        Input:
            Time x Batch x Channel during training
        Args:
        """
        output = super().forward(input)
        if self.kernel_size[0] > 1 and self.padding[0] > 0:
            # remove future timesteps added by padding
            output = output[:-self.padding[0], :, :]
        return output

