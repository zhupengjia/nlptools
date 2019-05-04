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

    def forward(self, query, key, value, mask=None, incremental_state=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        print("scores", scores.size(), self.__class__.__name__)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = self.softmax(scores)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


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

    def forward(self, query, key, value, mask=None, incremental_state=None):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from embed_dim => num_heads x head_dim
        print("ma1", query.size(), key.size(), value.size())
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        print("ma2", query.size(), key.size(), value.size())

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, incremental_state=incremental_state)
        print("ma3", x.size())

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        print("ma4", x.size())
        return self.output_linear(x)


