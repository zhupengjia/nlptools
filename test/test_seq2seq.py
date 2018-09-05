#!/usr/bin/env python

import sys, torch, numpy
from nlptools.text import Vocab
from nlptools.text.embedding import Embedding_Random
from nlptools.zoo.demodata.demo_seq import demo_seq
from nlptools.zoo.seq2seq.fconv_seq2seq import FConvSeq2Seq
from nlptools.zoo.seq2seq.transformer import Transformer
from nlptools.zoo.seq2seq.lstm_seq2seq import LSTMSeq2Seq

def fconv():
    inputs, prev_outputs, outputs, vocab = demo_seq() 

    vocab.embedding = Embedding_Random(dim = 8)

    model = FConvSeq2Seq(
                vocab, vocab,
                encoder_layers = [(32, 4)]*13,
                decoder_layers = [(32, 4)]*12,
                share_embed=True
            )

    model.train(inputs, prev_outputs, outputs, num_epoch=200, max_words=20)
   

def transformer():
    inputs, prev_outputs, outputs, vocab = demo_seq() 

    vocab.embedding = Embedding_Random(dim = 8)

    model = Transformer(
                vocab, vocab,
                share_embed=True
            )

    model.train(inputs, prev_outputs, outputs, num_epoch=200, max_words=20)


def lstm():
    inputs, prev_outputs, outputs, vocab = demo_seq() 

    vocab.embedding = Embedding_Random(dim = 8)

    model = FConvSeq2Seq(
                vocab, vocab,
                share_embed=True
            )

    model.train(inputs, prev_outputs, outputs, num_epoch=200, max_words=20)

if __name__ == '__main__':
    #fconv()
    #transformer()
    lstm()

