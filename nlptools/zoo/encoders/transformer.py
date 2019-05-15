#!/usr/bin/env python
import torch, math
import torch.nn as nn
from pytorch_pretrained_bert.modeling import gelu, BertLayerNorm, BertModel, BertConfig

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

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
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_probs = self.dropout(nn.Softmax(dim=-1)(scores))
        context_layer = torch.matmul(attn_probs, v["v"])

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if incre_state is not None:
            context_layer = context_layer[:, prev_size:, :]

        return self.output_linear(context_layer)


class TransformerEncoder(BertModel):
    """
        Transformer Encoder, call BertModel directly
    """
    def __init__(self, vocab_size=30522, num_hidden_layers=12, num_attention_heads=12,
                 max_position_embeddings=512, intermediate_size=3072, hidden_size=768, dropout=0.1):
        config = BertConfig(vocab_size_or_config_json_file=vocab_size,
                            num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads,
                            max_position_embeddings=max_position_embeddings,
                            intermediate_size=intermediate_size,
                            hidden_size=hidden_size,
                            attention_probs_dropout_prob=dropout,
                            hidden_act="gelu",
                            hidden_dropout_prob=dropout,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12,
                            type_vocab_size=2)
        super(TransformerEncoder, self).__init__(config=config)


class TransformerDecoder(nn.Module):
    """
        transformer decoder
        worked with pretrained bert model from pytorch_pretrained_bert
    """

    def __init__(self, bert_embedding, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024, dropout=0.1, shared_embed=True):
        super().__init__()
        self.dropout = dropout
        
        self.word_embedding = bert_embedding.word_embeddings
        self.position_embedding = bert_embedding.position_embeddings
        self.layer_norm = bert_embedding.LayerNorm
        
        num_embeddings = self.word_embedding.num_embeddings
        embedding_dim = self.word_embedding.embedding_dim
        self.num_attention_heads = num_attention_heads

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_dim=embedding_dim, 
                               attention_heads = num_attention_heads,
                               dropout = dropout,
                               intermediate_size = intermediate_size)
            for i in range(num_hidden_layers)
        ])
        
        self.fc3 = nn.Linear(embedding_dim, num_embeddings, bias=False)
        if shared_embed:
            self.fc3.weight = self.word_embedding.weight

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask,
                time_step=0, incre_state=None):
        # embed tokens and positions
        word_embeddings = self.word_embedding(prev_output_tokens)
        position_ids = torch.arange(time_step, prev_output_tokens.size(1) + time_step,
                                    dtype=torch.long, device=prev_output_tokens.device)
        position_embeddings = self.position_embedding(position_ids)
        x = word_embeddings + position_embeddings
        x = self.layer_norm(x)
        
        #masks
        encoder_padding_mask = encoder_padding_mask.unsqueeze(1).unsqueeze(2)

        self_attn_mask = self.buffered_future_mask(x) if incre_state is None else None

        # print("transformer_embedding", x.shape)
        # print("encoder_padding_mask", encoder_padding_mask.shape)

        #decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incre_state=incre_state
            )

        # project back to size of vocabulary
        x = self.fc3(x)
        return x

    def buffered_future_mask(self, x):
        dim = x.size(1)
        mask = x.new(dim, dim).fill_(1).byte()
        return ~torch.triu(mask, 1)


    def reorder_incremental_state(self, incre_state, order):
        if incre_state is None:
            return
        for k1 in incre_state:
            for k2 in incre_state[k1]:
                incre_state[k1][k2] = incre_state[k1][k2][order, :, :]

class ResidualLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = BertLayerNorm(hidden_size, eps=1e-12)
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

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class TransformerDecoderLayer(nn.Module):
    """
    Decoder layer block.
    """

    def __init__(self, embed_dim, attention_heads, intermediate_size, dropout=0.1, no_encoder_attn=False):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, attention_heads,dropout=dropout)

        if no_encoder_attn:
            self.encoder_attn = None
        else:
            self.encoder_attn = MultiheadAttention(embed_dim, attention_heads,dropout=dropout)
            self.encoder_attn_residual = ResidualLayer(hidden_size=embed_dim, dropout=dropout)

        self.self_attn_residual = ResidualLayer(hidden_size=embed_dim, dropout=dropout)
        self.output_residual = ResidualLayer(hidden_size=embed_dim, dropout=dropout)
        self.feed_forward = FeedForward(embed_dim, intermediate_size, dropout) 


    def forward(self, x, encoder_out, encoder_padding_mask=None, self_attn_mask=None, incre_state=None):
        '''
        encoded output of shape (batch, src_len, embed_dim)
        '''
        # print("decoder_layer_input", x.shape)
        x = self.self_attn_residual(x, lambda _x: self.self_attn(
            _x, _x, _x, mask=self_attn_mask, incre_state=incre_state,
            incre_keys=["q","k","v"]))

        # print("self_attn", x.shape)

        if self.encoder_attn is not None:
            x = self.encoder_attn_residual(x, lambda _x: self.encoder_attn(
                _x, encoder_out, encoder_out,
                mask=encoder_padding_mask, incre_state=incre_state, incre_keys=["q"]))
            # print("encoder_attn", x.shape)

        x = self.output_residual(x, self.feed_forward)
        # print("output", x.shape)

        return x

