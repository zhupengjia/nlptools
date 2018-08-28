#!/usr/bin/env python
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .classifier_base import ClassifierBase

class Classifier(ClassifierBase):
    def __init__(self, target_size, vocab, pretrained_embed=True, cnn_kernel_num=20, cnn_kernel_size=5, pool_size=2, dropout=0.8, max_seq_len=16, device='cpu'):
        super().__init__(vocab, pretrained_embed, device)
    
        self.embedding = nn.Embedding(num_embeddings = self.vocab.vocab_size, \
                embedding_dim = self.vocab.emb_ins.vec_len, \
                padding_idx = self.vocab._id_PAD)

        if pretrained_embed:
            self.embedding.weight.data = torch.FloatTensor(self.vocab.dense_vectors())
        #self.embedding.weight.requires_grad = False
        self.conv = nn.Conv2d(in_channels = 1, \
                out_channels = self.cnn_kernel_num, \
                kernel_size = (self.cnn_kernel_size, self.vocab.emb_ins.vec_len),\
                padding = 0)
        self.pool = nn.MaxPool1d(self.pool_size)
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.cnn_kernel_num + int((self.max_seq_len - (self.cnn_kernel_size-1))/self.pool_size), target_size)

    def conv_and_pool(self, x):
        x = self.conv(x)
        x = F.relu(x) 
        x = x.squeeze(3)
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, question):
        q1 = self.embedding(question)
        q1 = q1.unsqueeze(1)
        q1 = self.conv_and_pool(q1)

        M_rowsum = q1.sum(dim=1)
        M_colsum = q1.sum(dim=2)
        M_att = torch.cat((M_rowsum, M_colsum), 1)
        
        out = self.fc1(M_att)
        return out 


