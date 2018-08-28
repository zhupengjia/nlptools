#!/usr/bin/env python
import torch
import torch.nn as nn

class TransformerClassifier(ClassifierBase):
    def __init__(self, target_size, vocab, pretrained_embed=True, layers=3, attention_heads=4, ffn_embed_dim=512, dropout=0.1, device='cpu'):
        super().__init__(vocab, pretrained_embed, device)
        
        self.encoder = TransformerEncoder(
                    vocab = self.encoder_vocab,
                    pretrained_embed = pretrained_embed,
                    layers = encoder_layers,
                    attention_heads = encoder_attention_heads,
                    ffn_embed_dim = encoder_ffn_embed_dim,
                    dropout = dropout
                ) 

        self.fc = nn.Linear(in_feature=ffn_embed_dim, out_feature=target_size)

    def forward(self, sentence):
        sentence_emb = self.encoder(sentence)
        out = self.fc(sentence_emb)
        return out

