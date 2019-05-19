#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .attention import MultiheadAttention


class LSTMEncoder(nn.Module):
    """
        LSTM Encoder
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=1,
                 bidirectional=True, dropout=0.1, **args):
        super().__init__()
        self.config = {"vocab_size": vocab_size,
                       "hidden_size": hidden_size,
                       "num_hidden_layers": num_hidden_layers,
                       "bidirectional": bidirectional}
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_hidden_layers, dropout=dropout if num_hidden_layers > 1 else 0,
                            bidirectional=bidirectional, batch_first=True)

        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.linear_out_x = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_size)
        self.linear_out_ht = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_size)
        self.linear_out_ct = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, **args):
        length = attention_mask.sum(dim=1)
        length, new_order = torch.sort(length, 0, descending=True)
        _, un_order = new_order.sort(0)

        bsz = input_ids.size(0)
        x = self.embeddings(input_ids)
        x = x[new_order, :, :]
        x = pack_padded_sequence(x, length, batch_first=True)
        x, (ht, ct) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[un_order]
        ht = ht[:, un_order, :]
        ct = ct[:, un_order, :]

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_hidden_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_hidden_layers, bsz, -1)
            ht = combine_bidir(ht)
            ct = combine_bidir(ct)
        x = self.linear_out_x(x) 
        ht = self.linear_out_ht(ht) 
        ct = self.linear_out_ct(ct) 

        return x, (ht, ct)


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim, output_dim):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, source_dim)
        self.output_linear = nn.Linear(input_dim+source_dim, output_dim)

    def forward(self, data, attention_data, mask):
        x = self.input_linear(data)
        x = x.unsqueeze(1)
        scores = torch.matmul(x, attention_data.transpose(-1,-2))
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        scores = nn.Softmax(dim=-1)(scores)
        x = self.output_linear(x, dim=1)
        print(scores, scores.shape)
        return x, scores
        x = (attn_scores.unsqueeze(2) * encoder_out).sum(dim=0)
        return x, attn_scores


class LSTMDecoder(nn.Module):
    """
        LSTM Decoder
    """
    def __init__(self, word_embedding, num_hidden_layers=1, attention=True, num_attention_heads=8, dropout=0.1, shared_embed=True):
        super().__init__()
        self.config = {"num_attention_heads": num_attention_heads,
                       "num_hidden_layers": num_hidden_layers,
                       "shared_embed": shared_embed,
                       "attention": attention
                      }
        self.word_embedding = word_embedding

        self.hidden_size = self.word_embedding.embedding_dim
        self.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size
            )
            for layer in range(num_hidden_layers)
        ])
        if attention:
            self.attention = MultiheadAttention(self.hidden_size, num_attention_heads, dropout=dropout)
        else:
            self.attention = None
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if shared_embed:
            self.output_linear.weight = self.word_embedding.weight


    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask=None,
                encoder_hidden=None, incre_state=None, **args):
        bsz, seqlen = prev_output_tokens.size()

        embeddings = self.word_embedding(prev_output_tokens)

        obj_id = id(self)
        if incre_state is not None and obj_id in incre_state:
            prev_hiddens, prev_cells = incre_state[obj_id]
        elif encoder_hidden is not None:
            prev_hiddens = encoder_hidden[0][:self.num_hidden_layers, :, :]
            prev_cells = encoder_hidden[1][:self.num_hidden_layers, :, :]
        else:
            prev_hiddens = prev_out_tokens.new_zeros(self.num_hidden_layers, bsz, self.hidden_size)
            prev_cells = prev_out_tokens.new_zeros(self.num_hidden_layers, bsz, self.hidden_size)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1).unsqueeze(2)

        outs = []
        for j in range(seqlen):
            x = embeddings[:, j, :]
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(x, (prev_hiddens[i], prev_cells[i]))
                x = self.dropout(hidden)
                prev_hiddens[i], prev_cells[i] = hidden, cell
                # attention
                x = self.attention(x.unsqueeze(1), encoder_out, encoder_out, mask=encoder_padding_mask)
                x = x.squeeze(1)
            outs.append(x.unsqueeze(1))

        if incre_state is not None:
            incre_state[obj_id] = prev_hiddens, prev_cells
        
        
        outs = torch.cat(outs, dim=1)
        outs = self.output_linear(outs)
        return outs

