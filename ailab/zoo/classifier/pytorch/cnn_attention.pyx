#!/usr/bin/env python
import torch, sys
import torch.nn.functional as F
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, cfg, vocab=None):
        super().__init__()
        self.cfg = {'cnn_kernel_num':20,\
                    'cnn_kernel_size':5,\
                    'pool_size':2,\
                    'dropout':0.8,\
                    'max_seq_len':16\
                }
        for k in cfg:
            if k in self.cfg:
                self.cfg[k] = cfg[k]
        self.vocab = vocab
    
    def network(self, target_size):
        self.embedding = nn.Embedding(num_embeddings = self.vocab.vocab_size, \
                embedding_dim = self.vocab.emb_ins.vec_len, \
                padding_idx = self.vocab._id_PAD)

        self.embedding.weight.data = torch.FloatTensor(self.vocab.dense_vectors())
        #self.embedding.weight.requires_grad = False
        self.conv = nn.Conv2d(in_channels = 1, \
                out_channels = self.cfg['cnn_kernel_num'], \
                kernel_size = (self.cfg['cnn_kernel_size'], self.vocab.emb_ins.vec_len),\
                padding = 0)
        self.pool = nn.MaxPool1d(self.cfg['pool_size'])
        self.dropout = nn.Dropout(self.cfg['dropout'])
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num'] + int((self.cfg['max_seq_len'] - (self.cfg['cnn_kernel_size']-1))/self.cfg['pool_size']), target_size)

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

        #self attention
        M_rowsum = q1.sum(dim=1)
        M_colsum = q1.sum(dim=2)
        M_att = torch.cat((M_rowsum, M_colsum), 1)
        
        out = self.fc1(M_att)
        return out 


