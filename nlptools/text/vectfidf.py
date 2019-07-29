#!/usr/bin/env python3
import numpy, os, pandas, re
from sklearn.metrics.pairwise import pairwise_distances
from .tfidf import TFIDF
from .vocab import Vocab

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class VecTFIDF(TFIDF):
    '''
        Modified TF-IDF algorithm with wordvector

        Input:
            - vocab: instance of .vocab, default is None
            - freqwords_path: path of freqword list, will force set the count of each word in the list to a large number
            - cached_index: path of cached index file
    '''
    def __init__(self, vocab = None, cached_index = '', freqwords_path=''):
        self.vocab = vocab 
        if self.vocab is None: self.vocab = Vocab()
        self.freqwords_path = freqwords_path
        self.distance_metric = 'cosine'
        self.scorelimit = 0.6
        self.freqwords = {}
        self.__loadFreqwords(self.freqwords_path)
        TFIDF.__init__(self, vocab_size=self.vocab.vocab_size, cached_index=cached_index)

    def __loadFreqwords(self, freqwords_path=None):
        if os.path.exists(freqwords_path):
            print('load freqword path', freqwords_path)
            with open(freqwords_path) as f:
                for l in f.readlines():
                    for w in re.split("\s", l):
                        w = w.strip()
                        if len(w) < 1: continue
                        wordid = self.vocab.word2id(w)
                        self.freqwords[wordid] = 0


    def n_sim(self, word_ids, sentence_ids):
        '''
            Get the sum of similarities of each id in a token_id list with a sentence(another token_id list)

            Input:
                - word_ids: a token id list
                - sentence_ids: a token id list for sentence

            output:
                - 1d numpy array
        '''
        if len(sentence_ids) < 1:
            return 0
        if isinstance(word_ids, int):
            word_ids = [word_ids]
        sentence_vec = self.vocab.ids2vec(sentence_ids)
        word_vec = self.vocab.ids2vec(word_ids)
        score = 1/(1.+pairwise_distances(word_vec, sentence_vec, metric = self.distance_metric))
        score[score < self.scorelimit] = 0
        return score.sum(axis = 1)


    def vectf(self, word_ids, sentence_ids):
        '''
            calculate term frequencies for each word in a token id list

            Input:
                - word_ids: a token id list
                - sentence_ids: a token id list for sentence

            output:
                - 1d numpy array
        '''
        if len(sentence_ids) == 0:
            return 0
        return self.n_sim(word_ids, sentence_ids)/len(sentence_ids)
   
    def vectfidf(self, word_ids, sentence_ids, word_idfs):
        '''
            calculate tfidf for each id in a token list

            Input:
                - word_ids: token ids
                - sentence_ids: token ids in a sentence
                - word_idfs: idf for token ids

            Output:
                tfidf list
        '''
        tf = self.vectf(word_ids, sentence_ids)
        idf = word_idfs[word_ids]
        return tf*idf

    #vec TF-IDF
    def search(self, word_ids, corpus_ids, topN=1, global_idfs=True):
        '''
            Calculate and sort search scores  via vectfidf

            Input:
                - word_ids: token ids
                - corpus_ids: a list of token ids as corpus
                - topN: int, return topN result, default is 1
                - global_idfs, bool, use global idfs calculated from large corpus when doing load_index or idfs from input corpus_ids, default is True

            Output:
                - [(score, tfidf), ...], ...

        '''
        corpus_ids = pandas.Series(corpus_ids)
        if self.word_idfs is not None and global_idfs:
            word_idfs = self.word_idfs
        else:
            count_matrix, word_idfs = self.load_index(corpus_ids, local_use=True)
        tfidf = corpus_ids.apply(lambda x: self.vectfidf(word_ids, x, word_idfs).sum()).as_matrix()
        
        scores = numpy.argsort(tfidf)[::-1]
        scores = [int(s) for s in scores[:topN]]
        tfidf = tfidf[scores]
        return list(zip(scores, tfidf))


