#!/usr/bin/env python
'''
    Multihead attention
'''

import torch, math
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None, incre_state=None, incre_keys=None):
        v = {"q":q, "k":k, "v":v}

        # concat incremental state
        obj_id = id(self)
        if incre_state is not None:
            if obj_id in incre_state:
                prev_size = incre_state[obj_id]['q'].size(1)
                for k in incre_keys:
                    v[k] = torch.cat((incre_state[obj_id][k], v[k]), dim=1)
            else:
                incre_state[obj_id] = {}
                prev_size = 0
            for k in incre_keys:
                incre_state[obj_id][k] = v[k]

        # Do all the linear projections in batch from embed_dim => num_heads x head_dim
        v["q"],v["k"],v["v"] = [self.transpose_for_scores(l(x))
                                for l, x in zip(self.linear_layers,
                                                (v["q"], v["k"], v["v"]))]

        # Apply attention on all the projected vectors in batch.
        scores = torch.matmul(v["q"], v["k"].transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e3)

        attn_probs = self.dropout(nn.Softmax(dim=-1)(scores))
        context_layer = torch.matmul(attn_probs, v["v"])

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if incre_state is not None:
            context_layer = context_layer[:, prev_size:, :]

        return self.output_linear(context_layer)
