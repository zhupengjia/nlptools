#!/usr/bin/env python
from ailab.utils import zload, zdump
from ailab.text import Segment, Embedding, Vocab
import os, pandas, sys, numpy, math, torch
from torch.autograd import Variable

class DataReader(object):
    def __init__(self, cfg, gpu=False, vocab=None):
        self.cfg = cfg
        self.gpu = gpu
        self.seg = Segment(self.cfg)
        self.emb = Embedding(self.cfg)
        if vocab is None:
            self.vocab = Vocab(self.cfg, self.seg, self.emb)
            self.vocab.addBE()
        else:
            self.vocab = vocab

    def reduce_vocab(self):
        self.vocab.reduce_vocab()
    
    def __call__(self, sentences, targets=None, shuffle=False):
        ids_return = numpy.ones((len(sentences), self.cfg['max_seq_len']), 'int') * self.vocab._id_PAD
        for i, s in enumerate(sentences):
            sentence_ids = self.vocab.sentence2id(s, update=False)
            char_ids = [self.vocab.add_word(c.strip()) for c in s if len(c.strip())>0]
            ids = sentence_ids + char_ids
            qlen = min(len(ids), self.cfg['max_seq_len'])
            ids_return[i][:qlen] = ids[:qlen]
        if shuffle:
            shuffle_ids = numpy.arange(len(sentences))
            numpy.random.shuffle(shuffle_ids)
            ids_return = ids_return[shuffle_ids]
        if self.gpu:
            X = Variable(torch.LongTensor(ids_return).cuda(self.gpu-1))
        else:
            X = Variable(torch.LongTensor(ids_return))
        if targets is None:
            return X
        else:
            targets = numpy.array(targets, 'int')
            if shuffle: targets = targets[shuffle_ids]
            if self.gpu:
                Y = Variable(torch.LongTensor(targets).cuda(self.gpu-1))
            else:
                Y = Variable(torch.LongTensor(targets))
            return X, Y



