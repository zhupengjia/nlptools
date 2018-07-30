#!/usr/bin/env python

from .vocab import Vocab

'''
    
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class BytePair(Vocab):
    def __init__(self, **args):
        '''
            Byte Pair Encoding (BPE) vocabulary for rare word, see https://arxiv.org/abs/1508.07909. Only used for latin languages

            Input:
                - any available parameters in nlptools.text.vocab
        '''
        super().__init__(**args)


    def learn(self):
        '''
            learn bpe
        '''
        pass

    
    def apply(self):
        '''
            apply bpe
        '''
        pass

