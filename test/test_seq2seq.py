#!/usr/bin/env python

import sys, torch, numpy
from nlptools.text import Vocab
from nlptools.text.embedding import Embedding_Random
from nlptools.zoo.demodata.demo_seq import demo_seq
from nlptools.zoo.tagging.fconv_seq2seq import FConvSeq2Seq

def main():
    inputs, prev_outputs, outputs, vocab = demo_seq() 

    vocab.embedding = Embedding_Random(dim = 8)

    model = FConvSeq2Seq(
                vocab, vocab,
                encoder_layers = [(32, 4)]*13,
                decoder_layers = [(32, 4)]*12,
            )


    model.train(inputs, prev_outputs, outputs, num_epoch=200, max_words=20)
   

if __name__ == '__main__':
    main()


