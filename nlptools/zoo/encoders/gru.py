#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import torch, h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUEncoder(nn.Module):
    """
        GRU Encoder
    """
    def __init__(self, vocab_size, pretrained_embedding=None, hidden_size=768,
                 intermediate_size=1024, num_hidden_layers=1,
                 bidirectional=True, dropout=0.1, **args):
        super().__init__()
        self.config = {"vocab_size": vocab_size,
                       "hidden_size": hidden_size,
                       "intermediate_size": intermediate_size,
                       "num_hidden_layers": num_hidden_layers,
                       "bidirectional": bidirectional}
        if pretrained_embedding:
            with h5py.File(pretrained_embedding, 'r') as h5file:
                self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(h5file["word2vec"]))
        else:
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=intermediate_size,
                            num_layers=num_hidden_layers, dropout=dropout if num_hidden_layers > 1 else 0,
                            bidirectional=bidirectional, batch_first=True)

        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.linear_out_x = nn.Linear(intermediate_size*2 if bidirectional else intermediate_size, hidden_size)
        self.linear_out_hidden = nn.Linear(intermediate_size*2 if bidirectional else intermediate_size, hidden_size)

    def forward(self, input_ids, attention_mask, **args):
        max_seq_len = input_ids.size(1)
        length = attention_mask.sum(dim=1)
        length, new_order = torch.sort(length, 0, descending=True)
        _, un_order = new_order.sort(0)

        bsz = input_ids.size(0)
        x = self.embeddings(input_ids)
        x = x[new_order, :, :]
        x = pack_padded_sequence(x, length, batch_first=True)
        x, hidden = self.gru(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        
        if x.size(1) < max_seq_len:
            x = F.pad(x, (0,0,0,max_seq_len-x.size(1),0,0), "constant")

        x = x[un_order]
        hidden = hidden[:, un_order, :]

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_hidden_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_hidden_layers, bsz, -1)
            hidden = combine_bidir(hidden)
        x = self.linear_out_x(x) 
        hidden = self.linear_out_hidden(hidden) 

        return x, hidden


class GRUDecoder(nn.Module):
    """
        GRU Decoder
    """
    def __init__(self, word_embedding, num_hidden_layers=1, attention=True,
                 intermediate_size=1024, max_seq_len=20, dropout=0.1, shared_embed=True):
        super().__init__()
        self.config = {"num_hidden_layers": num_hidden_layers,
                       "intermediate_size": intermediate_size,
                       "shared_embed": shared_embed,
                       "max_seq_len": max_seq_len,
                       "attention": attention
                      }
        self.word_embedding = word_embedding

        self.hidden_size = self.word_embedding.embedding_dim
        self.num_embeddings = self.word_embedding.num_embeddings
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.intermediate_size,
                            num_layers=num_hidden_layers,
                            dropout=dropout if num_hidden_layers > 1 else 0, batch_first=True)

        if attention:
            self.attn = nn.Linear(self.hidden_size+self.intermediate_size,  max_seq_len)
            self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.attn = None
            self.attn_combine = None
        self.dropout = nn.Dropout(dropout)
        if self.intermediate_size != self.hidden_size:
            self.hidden_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        else:
            self.hidden_proj = None

        self.intermediate_linear = nn.Linear(self.intermediate_size, self.hidden_size)
        self.output_linear = nn.Linear(self.hidden_size, self.num_embeddings, bias=False)
        if shared_embed:
            self.output_linear.weight = self.word_embedding.weight


    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask=None,
                encoder_hidden=None, incre_state=None, **args):
        bsz, seqlen = prev_output_tokens.size()

        out = self.word_embedding(prev_output_tokens)

        obj_id = id(self)
        if incre_state is not None and obj_id in incre_state:
            hidden = incre_state[obj_id]
        elif encoder_hidden is not None:
            hidden = encoder_hidden[:self.num_hidden_layers, :, :]
            if self.hidden_proj is not None:
                hidden = self.hidden_proj(hidden)
        else:
            hidden = encoder_out.new_zeros(self.num_hidden_layers, bsz, self.intermediate_size)

        if self.attn is not None:
            attn = hidden[0].unsqueeze(1).expand(-1, seqlen, -1)
            attn = torch.cat((out, attn), 2)
            score = nn.Softmax(dim=2)(self.attn(attn))

            if encoder_padding_mask is not None:
                encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
                score = score.masked_fill(encoder_padding_mask == 0, -1e9)

            attn = torch.matmul(score, encoder_out)
            out = torch.cat((out, attn), 2)
            out = self.attn_combine(out)

        out, hidden = self.gru(out, hidden)

        if incre_state is not None:
            incre_state[obj_id] = hidden

        out = self.intermediate_linear(out)
        out = self.output_linear(out)
        return out

    def reorder_incremental_state(self, incre_state, order):
        if incre_state is None:
            return
        for k1 in incre_state:
            incre_state[k1] = incre_state[k1][:, order, :]

