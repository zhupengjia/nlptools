#!/usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.adaptive_softmax import AdaptiveSoftmax
from ..modules.linearized_convolution import LinearizedConv1d
from .decoder_base import Decoder_Base
from .fconv_encoder import Embedding, extend_conv_spec, PositionalEmbedding, Linear


class FConvDecoder(Decoder_Base):
    """Convolutional decoder"""

    def __init__(
        self, vocab, predtrained_embed=True, out_embed_dim=256,
        max_positions=1024, convolutions=((512, 3),) * 20, attention=True,
        dropout=0.1, share_embed=False, positional_embeddings=True,
        adaptive_softmax_cutoff=None, normalization_constant=0.5,
        left_pad=False,
    ):
        super().__init__(vocab)
        self.dropout = dropout
        self.normalization_constant = normalization_constant
        self.left_pad = left_pad

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = vocab.vocab_size
        padding_idx = vocab.PAD_ID
        embed_dim = vocab.embedding_dim
        
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if pretrained_embed:
            self.embed_tokens.weight.data = torch.FloatTensor(vocab.dense_vectors())

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=self.left_pad,
        ) if positional_embeddings else None

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(AttentionLayer(out_channels, embed_dim, self.normalization_constant)
                                  if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, in_channels, adaptive_softmax_cutoff, dropout=dropout)
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, \
                    "Shared embed weights implies same dimensions " \
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out_dict=None):
        if encoder_out_dict is not None:
            encoder_out = encoder_out_dict['encoder_out']
            encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens)
        else:
            pos_embed = 0

        x = self.embed_tokens(prev_output_tokens)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = x.transpose(0, 1)

                x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)

                x = x.transpose(0, 1)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(self.normalization_constant)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            return out.exp_() if not log_probs else out
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions() if self.embed_positions is not None else float('inf')


    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs.
        """
        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)
        return result


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, normalization_constant=0.5, bmm=None):
        super().__init__()
        self.normalization_constant = normalization_constant
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(self.normalization_constant)
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            ).type_as(x)  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(self.normalization_constant)
        return x, attn_scores




