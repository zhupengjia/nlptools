#!/usr/bin/env python3
import numpy
from sklearn.metrics.pairwise import cosine_distances
#from ..utils import zload, zdump

class VecTFIDF:
    def __init__(self, cfg, vocab_ins=None):
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
    
    def tf(self, word_id, sentence_ids):
        return self.n_count(word_id, sentence_ids)/len(sentence_ids)

    def n_containing(self, word_id, corpus_ids):
        return sum(1 for sentence_ids in corpus_ids if self.n_count(word_id, sentence_ids)>=1 )

    def idf(self, word_id, corpus_ids):
        return numpy.log(len(corpus_ids)/(1. + self.n_containing(word_id, corpus_ids)))

    def tfidf(self, word_id, sentence_ids, corpus_ids):
        return self.tf(word_id, sentence_ids) * self.idf(word_id, corpus_ids)


