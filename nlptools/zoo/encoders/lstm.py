#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
        LSTM Encoder
    """
    def __init__(self, vocab_size, embed_dim, pad_id, hidden_size, num_layers, bidirectional=True,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_id)
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)


    def forward(self, src_tokens, src_padding_mask):
        x = self.embedding(x)



class LSTMDecoder(nn.Module):
    """
        LSTM Decoder
    """
    def __init__(self, word_embedding, num_layers):
        self.word_embedding = word_embedding
        self.dropout = dropout

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask, time_step=0, incre_state=None):
        pass

