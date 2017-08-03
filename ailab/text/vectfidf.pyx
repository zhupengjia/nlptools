#!/usr/bin/env python3
import numpy, sys, multiprocessing, time
from sklearn.metrics.pairwise import cosine_distances
from functools import partial
from ailab.utils import zload, zdump, n_count

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins=None):
        self.cfg = cfg
        self.vocab = vocab_ins 
        self.scorelimit = 0.6

    def n_sim(self, word_id, sentence_ids):
        if len(sentence_ids) < 1:
            return 0
        sentence_ids, idcounts = numpy.unique(sentence_ids, return_counts=True)
        sentence_vec = self.vocab.senid2vec(sentence_ids)
        word_vec = self.vocab.word2vec(word_id)
        score = 1/(1.+cosine_distances([word_vec], sentence_vec)[0])
        score[score < self.scorelimit] = 0
        if idcounts is not None:
            score = score * idcounts
        return score.sum()

    #calculate tf
    def tf(self, word_id, sentence_ids):
        return self.n_sim(word_id, sentence_ids)/len(sentence_ids)
    
    #build index for each word in corpus
    def train_index(self, corpus_ids):
        self.index_word2doc = {}
        self.len_corpus = len(corpus_ids)
        all_ids_1 = list(set([item for sublist in corpus_ids for item in sublist]))
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)

        for i in all_ids_1:
            t1 = time.time()
            func_n_count = partial(n_count, i)
            ncounts = numpy.array(pool.map(func_n_count, corpus_ids), 'int32')
            self.index_word2doc[all_ids_1[i]] = {j:ncounts[j] for j in ncounts.nonzero()[0]}
            del func_n_count
            t2 = time.time()
            print('building index, ', i, len(all_ids_1), t2-t1)

        if 'cached_index' in self.cfg:
            zdump((self.index_word2doc, self.len_corpus), self.cfg['cached_index'])

    def load_index(self):
        self.index_word2doc, self.len_corpus = zload(self.cfg['cached_index'])

    def n_containing(self, word_id):
        if not word_id in self.index_word2doc:
            return 0
        return len([x for x in self.index_word2doc[word_id] if x>=1])

    def idf(self, word_id):
        return numpy.log(self.len_corpus/(1. + self.n_containing(word_id)))

    def tfidf(self, word_id, sentence_ids):
        return self.tf(word_id, sentence_ids) * self.idf(word_id)

    #def search(self, word_ids, sentence_ids):

    


