#!/usr/bin/env python
import torch
import torch.nn as nn
from .classifier_base import ClassifierBase
from ..encoders.transformer_encoder import TransformerEncoder


class TransformerClassifier(ClassifierBase):
    def __init__(self, target_size, vocab, pretrained_embed=True, layers=3, attention_heads=4, ffn_embed_dim=512, dropout=0.1, device='cpu'):
        super().__init__(vocab, pretrained_embed, device)
        
        self.encoder = TransformerEncoder(
                    vocab = vocab,
                    pretrained_embed = pretrained_embed,
                    layers = layers,
                    attention_heads = attention_heads,
                    ffn_embed_dim = ffn_embed_dim,
                    dropout = dropout
                ) 

        self.fc = nn.Linear(in_features=vocab.embedding_dim, out_features=target_size)

    def forward(self, sentence):
        x = self.encoder(sentence)['encoder_out']

        x = x.sum(dim=1)

        out = self.fc(x)
        return out

