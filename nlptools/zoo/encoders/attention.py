#!/usr/bin/env python
import math
import torch.nn as nn
import torch


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = self.softmax(scores)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, v), p_attn


class MultiheadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None, incre_state=None, incre_keys=None):
        batch_size = q.size(0)
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
        v["q"],v["k"],v["v"] =\
                [l(x).view(batch_size,
                           -1,
                           self.num_heads,
                           self.head_dim).transpose(1, 2)\
                 for l, x in zip(self.linear_layers,
                                 (v["q"], v["k"], v["v"]))]

        # Apply attention on all the projected vectors in batch.
        x, attn = self.attention(v["q"], v["k"], v["v"], mask=mask)
        

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if incre_state is not None:
            x = x[:, prev_size:, :] 

        return self.output_linear(x)


