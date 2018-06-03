#!/usr/bin/env python
import os, operator, sys, numpy
from .embedding import Embedding_File
from ..utils import zload, distance2similarity

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Synonyms:
    '''
        Get synonyms using annoy, via word vectors

        Input:
            - embedding: text.embedding object, for word2vec source
            - synonyms_path: path for synonyms index. If not exists, will generate from embedding.
            - w2v_word2idx: path for word-index bidict mapping
            - synonyms_filter: similarity score filter for finding synonyms, default is 0.5
            - synonyms_max: max number of synonyms, default is 1000

        Special usage:
            - __call__: get synonyms for word
                - input: 
                    - word: string
                    - Nlimit: number limit of returned synonyms, default is 1000
                    - scorelimit: similarity limit of returned synonyms, default is 0.5

    '''

    def __init__(self, embedding, synonyms_path='', w2v_word2idx=''):
        self.embedding = embedding
        self._load_index()


    def _load_index(self):
        '''
            Build or load synonyms index
        '''
        from annoy import AnnoyIndex
        self._search = AnnoyIndex(self.embedding.vec_len)
        self._word2idx = zload(self.w2v_word2idx)
        if os.path.exists(self.synonyms_path):
            self._search.load(self.synonyms_path)
        else:
            assert isinstance(self.embedding, Embedding_File), 'Word embedding must from file source'
            for word, wordid in sorted(self._word2idx.items(), key=operator.itemgetter(1)):
                if wordid % 10000 == 0 :
                    print('building synonyms index, {}'.format(wordid))
                self._search.add_item(wordid, self.embedding[word])
            self._search.build(10)
            if len(self.synonyms_path) > 0:
                self._search.save(self.synonyms_path)
            
            
    def __call__(self, word, Nlimit = 1000, scorelimit = 0.5):
        '''
            Looking for synonyms
            
            Input:
                - word: string
                - Nlimit: number limit of returned synonyms, default is 1000
                - scorelimit: similarity limit of returned synonyms, default is 0.5
        '''
        if word in self._word2idx:
            result, score = self._search.get_nns_by_item(self._word2idx[word], Nlimit, include_distances=True)
        else:
            result, score = self._search.get_nns_by_vector(self.embedding[word], Nlimit, include_distances=True)
        result = [self._word2idx.inv[r] for r in result]
        score = distance2similarity(numpy.array(score))
        N_keep = len(score[score > scorelimit])
        result = result[:N_keep]
        score = score[:N_keep]
        return result, score



