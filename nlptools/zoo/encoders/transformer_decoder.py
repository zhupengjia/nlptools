#!/usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from fairseq.modules import MultiheadAttention

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class TransformerDecoder(FairseqIncrementalDecoder):
    """
        Transformer decoder. Modified from Fairseq
        Worked with pretrained BERT model from pytorch_pretrained_bert
    """

    def __init__(self, bert_embedding, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024, dropout=0.1, shared_embed=True):
        super().__init__(dictionary=None)
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

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        encoder_padding_mask = encoder_padding_mask.byte()
        encoder_padding_mask = ~encoder_padding_mask # for mask fill

        # decoder layers
        inner_states = [x]
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                encoder_padding_mask,
                incremental_state
            )
            inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc3(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        return  self.position_embedding.num_embeddings


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, layer_norm, embed_dim, attention_heads, ffn_embed_dim, dropout=0.1, normalize_before=False, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, attention_heads,
            dropout=dropout,
        )
        self.dropout = dropout
        self.normalize_before = normalize_before
        self.layer_norm = layer_norm

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, attention_heads,
                dropout=dropout,
            )

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)

        self.need_attn = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


