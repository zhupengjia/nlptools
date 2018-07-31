#!/usr/bin/env python

from .vocab import Vocab

'''
    
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class BytePair(Vocab):
    '''
        Learn Byte Pair Encoding, see https://arxiv.org/abs/1508.07909 
    '''
    EOW = '</w>'
    EOW_ID = 4

    def __init__(self, min_freq=2, **args):
        '''
            Byte Pair Encoding (BPE) vocabulary for rare word, see https://arxiv.org/abs/1508.07909. Only used for latin languages

            Input:
                - min_freq: int, Stop if no symbol pair has frequency >= min_freq, default is 2
                - any available parameters in nlptools.text.vocab
        '''
        super().__init__(**args)
        self._word_spec.append(self.EOW)
        self._id_spec.append(self.EOW_ID)


    def learn(self):
        '''
            learn bpe
        '''
        #word2tf = self._word2id
        pass

    
    def apply(self):
        '''
            apply bpe
        '''
        pass

