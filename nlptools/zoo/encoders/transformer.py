#!/usr/bin/env python
import torch, sys
import torch.nn as nn
from .attention import MultiheadAttention

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class TransformerDecoder(nn.Module):
    """
        Transformer decoder
        Worked with pretrained BERT model from pytorch_pretrained_bert
    """

    def __init__(self, bert_embedding, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024, dropout=0.1, shared_embed=True):
        super().__init__()
        self.dropout = dropout
       
        self.word_embedding = bert_embedding.word_embeddings
        self.position_embedding = bert_embedding.position_embeddings
        self.layer_norm = bert_embedding.LayerNorm

        num_embeddings = self.word_embedding.num_embeddings
        embedding_dim = self.word_embedding.embedding_dim

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_dim=embedding_dim, 
                               attention_heads = num_attention_heads,
                               dropout = dropout,
                               ffn_embed_dim = intermediate_size,
                               layer_norm=self.layer_norm)
            for i in range(num_hidden_layers)
        ])
        
        self.fc3 = nn.Linear(embedding_dim, num_embeddings, bias=False)
        if shared_embed:
            self.fc3.weight = self.word_embedding.weight

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask, time_step=0, incremental_state=None):
        # embed tokens and positions
        word_embeddings = self.word_embedding(prev_output_tokens)
        position_ids = torch.arange(time_step, prev_output_tokens.size(1), dtype=torch.long,
                                   device=prev_output_tokens.device)
        position_embeddings = self.position_embedding(position_ids)
        x = word_embeddings + position_embeddings
        x = self.layer_norm(x)

        encoder_padding_mask = encoder_padding_mask.unsqueeze(1).repeat(1, prev_output_tokens.size(1), 1).unsqueeze(1).byte()

        # decoder layers
        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        
        print("embedding", x.size(), encoder_out.size())
        print("mask", encoder_padding_mask.size(), self_attn_mask.size())
        
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )

        # project back to size of vocabulary
        x = self.fc3(x)
        return x

    def buffered_future_mask(self, x):
        dim = x.size(1)
        mask = x.new(dim, dim).fill_(1).byte()
        return ~torch.triu(mask, 1)


class ResidualLayer(nn.Module):
    def __init__(self, layer_norm, dropout):
        super().__init__()
        self.norm = layer_norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerDecoderLayer(nn.Module):
    """
    Decoder layer block.
    """

    def __init__(self, layer_norm, embed_dim, attention_heads, ffn_embed_dim, dropout=0.1, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, attention_heads,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = layer_norm

        if no_encoder_attn:
            self.encoder_attn = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, attention_heads,
                dropout=dropout,
            )
        
        self.feed_forward = FeedForward(self.embed_dim, attention_heads, dropout) 
        self.residual = ResidualLayer(layer_norm, dropout)        

    def forward(self, x, encoder_out, encoder_padding_mask, self_attn_mask=None, incremental_state=None):
        """
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x = self.residual(x, lambda _x: self.self_attn(
            _x, _x, _x, mask=self_attn_mask, incremental_state=incremental_state))

        if self.encoder_attn is not None:
            x = self.residual(x, lambda _x: self.encoder_attn(
                _x, encoder_out, encoder_out,
                mask=encoder_padding_mask, incremental_state=incremental_state))
        
        x = self.residual(x, self.feed_forward)
        return x



