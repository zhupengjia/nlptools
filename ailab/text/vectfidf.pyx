#!/usr/bin/env python3
import numpy
from functools import reduce
from sklearn.metrics.pairwise import cosine_distances
from ailab.utils import zload, zdump

class VecTFIDF:
    def __init__(self, cfg, vocab_ins=None):
        self.cfg = cfg
        self.vocab = vocab_ins 
        self.scorelimit = 0.6

    def n_count(self, word_id, sentence_ids):
        if len(sentence_ids) < 1:
            return float('inf')
        sentence_vec = self.vocab.senid2vec(sentence_ids)
        word_vec = self.vocab.word2vec(word_id)
        score = 1/(1.+cosine_distances([word_vec], sentence_vec)[0])
        score = score[score > self.scorelimit]
        return score.sum()
    
    #calculate tf
    def tf(self, word_id, sentence_ids):
        return self.n_count(word_id, sentence_ids)/len(sentence_ids)
    
    #build index for each word in corpus
    def train_index(self, corpus_ids):
        self.index_word2doc = {}
        self.len_corpus = len(corpus_ids)
        for i in set([item for sublist in corpus_ids for item in sublist]):
            self.index_word2doc[i] = {j:self.n_count(i, corpus_ids[j]) for j in range(len(corpus_ids))}
        if 'cached_index' in self.cfg:
            zdump((self.index_word2doc, self.len_corpus), self.cfg['cached_index'])

    def load_index(self):
        self.index_word2doc, self.len_corpus = zload(self.cfg['cached_index'])

    def n_containing(self, word_id):
        if not word_id in self.index_word2doc:
            return 0
        return len([x for x in self.index_word2doc[word_id].values() if x>=1])

    def idf(self, word_id):
        return numpy.log(self.len_corpus/(1. + self.n_containing(word_id)))

    def tfidf(self, word_id, sentence_ids):
        return self.tf(word_id, sentence_ids) * self.idf(word_id)


