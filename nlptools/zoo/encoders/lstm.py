#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
        LSTM Encoder
    """
    def __init__(self, vocab_size, hidden_size=768,
                 num_hidden_layers=1, bidirectional=True, dropout=0.1, **args):
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
        return x, (ht, ct)


class LSTMDecoder(nn.Module):
    """
        LSTM Decoder
    """
    def __init__(self, word_embedding, intermediate_size=3072, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        self.config = {"intermediate_size": intermediate_size,
                       "num_hidden_layers": num_hidden_layers
                      }
        self.word_embedding = word_embedding

        hidden_size = self.word_embedding.embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=hidden_size,
                hidden_size=intermediate_size,
            )
            for layer in range(num_hidden_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(intermediate_size, hidden_size)

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask,
                encoder_hidden=None, incre_state=None, **args):
        x = self.word_embedding(prev_output_tokens)
        obj_id = id(self)
        if incre_state is not None and obj_id in incre_state:
            prev_hiddens, prev_cells, input_feed = incre_state[obj_id]
        else:
            prev_hiddens = [encoder_hidden[0][i] for i in range(self.num_hidden_layers)]
            prev_cells = [encoder_hiddne[1][i] for i in range(self.num_hidden_layers)]
            pass

