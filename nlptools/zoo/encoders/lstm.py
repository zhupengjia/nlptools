#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import torch
import torch.nn as nn
from types import SimpleNamespace

class LSTMEncoder(nn.Module):
    """
        LSTM Encoder
    """
    def __init__(self, vocab_size, hidden_size=768, intermediate_size=3072,
                 num_hidden_layers=1, bidirectional=True, dropout=0.1, **args):
        super().__init__()
        self.config = {"vocab_size": vocab_size,
                       "hidden_size": hidden_size,
                       "intermediate_size": intermediate_size,
                       "num_hidden_layers": num_hidden_layers,
                       "bidirectional": bidirectional}
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=intermediate_size,
                            num_layers=num_hidden_layers, dropout=dropout if num_hidden_layers > 1 else 0,
                            bidirectional=bidirectional, batch_first=True)
        self.dropout = dropout
        self.output_linear = nn.Linear(intermediate_size, hidden_size)


    def forward(self, input_ids, attention_mask, **args):
        seq_lens = pass
        print(input_ids, input_ids.shape)
        print(attention_mask, attention_mask.shape)
        x = self.embeddings(input_ids)

        print(x)



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
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=intermediate_size, 
                            num_layers=num_hidden_layers, dropout=dropout if num_hidden_layers > 1 else 0,
                           batch_first=True)
        self.dropout = dropout
        self.output_linear = nn.Linear(intermediate_size, hidden_size)

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask, time_step=0,
                incre_state=None):
        x = self.embedding(prev_output_tokens)
        pass

