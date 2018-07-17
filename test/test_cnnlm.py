#!/usr/bin/env python
import sys, torch
from nlptools.text import Vocab
from nlptools.text.embedding import Embedding_Random
from nlptools.zoo.modules.bucket import demo_data, prepare_lm_data
from nlptools.zoo.tagging.fconv_lm import FConvLanguageModel


def main():

    '''
    PART I. Training
    '''
    
    inputs, targets, word_vocab, tag_vocab = demo_data()
    inputs, targets = prepare_lm_data(inputs)
    print('vocab_size: {}, tagset_size: {}'.format(len(word_vocab), len(tag_vocab)))
    print('data before trainig:')
    print(inputs)
    print(targets)


    word_vocab.embedding = Embedding_Random(dim = 8)


    model = FConvLanguageModel(
            word_vocab,
            decoder_layers = [(32, 4)]*13
    )
    
    
    model.train(inputs, targets, num_epoch=200, max_words=20)
   

if __name__ == '__main__':
    main()
