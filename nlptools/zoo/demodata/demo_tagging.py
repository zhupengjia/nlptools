#!/usr/bin/env python
import numpy
from nlptools.text import Vocab

def demo_tagging():
    
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("The dog".split(), ["DET", "NN"]),
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("Everybody read that book Everybody read that book".split(), ["NN", "V", "DET", "NN", "NN", "V", "DET", "NN"]),
        ("Everybody read that book The dog ate the apple".split(), ["NN", "V", "DET", "NN", "DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book Everybody read that book".split(), ["NN", "V", "DET", "NN", "NN", "V", "DET", "NN"])
    ]

    word_vocab = Vocab()
    tag_vocab = Vocab()
    
    inputs, tags = zip(*training_data)

    inputs = word_vocab(inputs, batch=True)
    tags = tag_vocab(tags, batch=True)
  
    word_vocab.reduce()
    tag_vocab.reduce()

    return inputs, tags, word_vocab, tag_vocab



if __name__ == '__main__':
    inputs, tags, word_vocab, tag_vocab = demo_tagging()
    print(inputs)
    print(tags)

