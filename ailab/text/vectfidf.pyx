#!/usr/bin/env python3
import numpy, sys, multiprocessing, time
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from ailab.utils import zload, zdump, n_count

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins=None):
        self.cfg = cfg
        self.vocab = vocab_ins 
        self.n_containing = numpy.vectorize(self.n_containing_id)
        self.distance_metric = 'cosince'
        self.scorelimit = 0.6

    def n_sim(self, word_ids, sentence_ids):
        if len(sentence_ids) < 1:
            return 0
        if isinstance(word_ids, int):
            word_ids = [word_ids]
        t1 = time.time()
        sentence_vec = self.vocab.senid2vec(sentence_ids)
        t2 = time.time()
        word_vec = self.vocab.senid2vec(word_ids)
        t3 = time.time()
        score = 1/(1.+pairwise_distances(word_vec, sentence_vec, metric = self.distance_metric))
        t4 = time.time()
        score[score < self.scorelimit] = 0
        t5 = time.time()
        score = score.sum(axis = 1)
        t6 = time.time()
        print(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
        return score

    #calculate tf
    def tf(self, word_ids, sentence_ids):
        return self.n_sim(word_ids, sentence_ids)/len(sentence_ids)
    
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

    def n_containing_id(self, word_id):
        if not word_id in self.index_word2doc:
            return 0
        return len([x for x in self.index_word2doc[word_id] if x>=1])

    def idf(self, word_ids):
        return numpy.log(self.len_corpus/(1. + self.n_containing(word_ids)))

    def tfidf(self, word_ids, sentence_ids):
        t1 = time.time()
        tf = self.tf(word_ids, sentence_ids)
        t2 = time.time()
        idf = self.idf(word_ids)
        t3 = time.time()
        print('tfidf', t2-t1, t3-t2)
        return tf*idf
        #return self.tf(word_ids, sentence_ids) * self.idf(word_ids)

    #def search(self, word_ids, sentence_ids):

    


