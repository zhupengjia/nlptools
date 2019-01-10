#!/usr/bin/env python
import numpy
from .vocab import Vocab 

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Ngrams(Vocab):
    '''
        vocab support ngrams
    '''

    def __init__(self, ngrams=1, **args):
        super().__init__(**args)
        self.ngrams = ngrams


    def words2id(self, tokens, batch=False):
        '''
            tokens to token ids
            
            Input:
                - tokens: token list
                - batch: if the input sequence is a batches, default is False

            Output:
                - list of ids

        '''
        if batch:
            return numpy.asarray([self.words2id(t) for t in tokens], dtype=numpy.object) 
        ids = {}
        ids[1] = [self.word2id(t) for t in tokens]
        ids[1] = numpy.array([i for i in ids[1] if i is not None], 'int')
        for i in range(2, self.ngrams+1):
            ids[i] = []
            for j in range(len(ids[1])-i+1):
                new_token = "".join(tokens[j:j+i])
                ids[i].append(self.word2id(new_token)) 
        return ids

    


