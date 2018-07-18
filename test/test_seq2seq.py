#!/usr/bin/env python

import sys, torch, numpy
from nlptools.text import Vocab
from nlptools.text.embedding import Embedding_Random
from nlptools.zoo.modules.bucket import demo_data, prepare_lm_data
from nlptools.zoo.tagging.fconv_seq2seq import FConvSeq2Seq

def main():
    training_data = [('what is the illuminati'.split(), 'a world wide conspiracy'.split()),
             ('what is the illuminatti'.split(), 'a secret society that has supposedly existed for centuries'.split()),
             ('what is vineland'.split(), 'vineland is a novel by thomas pynchon'.split()),
             ('what is illiminatus'.split(), 'alleged world wide conspiracy theory'.split()),
             ('who wrote vineland'.split(), 'thomas pynchon'.split()),
             ('who is bilbo baggins'.split(), "is a character in tolkein's lord of the rings".split()),
             ('who is geoffrey chaucer'.split(), 'chaucer is best known for his canterbury tales'.split()),
             ('who are the illuminati'.split(), 'who is geoffrey chaucer'.split()),
             ('who is piers anthony'.split(), 'author of canturbury tales'.split())]
    vocab = Vocab()
    
    inputs, outputs = zip(*training_data)
    inputs = vocab(inputs, batch=True)
    outputs = vocab(outputs, batch=True)

    inputs = [numpy.pad(i, (0, 1), 'constant', constant_values=Vocab.EOS_ID) for i in inputs]
    prev_outputs = [numpy.pad(o, (1, 0), 'constant', constant_values=Vocab.BOS_ID) for o in outputs]
    outputs = [numpy.pad(o, (0, 1), 'constant', constant_values=Vocab.EOS_ID) for o in outputs]


    vocab.reduce()

    vocab.embedding = Embedding_Random(dim = 8)

    model = FConvSeq2Seq(
                vocab, vocab,
                encoder_layers = [(32, 4)]*13,
                decoder_layers = [(32, 4)]*12,
            )


    model.train(inputs, targets, num_epoch=200, max_words=20)
   

if __name__ == '__main__':
    main()


