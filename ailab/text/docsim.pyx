#!/usr/bin/env python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine

class DocSim(object):
    def __init__(self, vocab):
        self.vocab = vocab


    def min_word_distance(self, sentence_id1, sentence_id2):
        vec1 = self.vocab.senid2vec(sentence_id1)
        vec2 = self.vocab.senid2vec(sentence_id2)
        return cosine_distances(vec1, vec2).min()
    
   
    def wcd_distance(self, sentence_id1, sentence_id2):
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = np.unique(sentence_id1 + sentence_id2)
        vectors = self.vocab.senid2vec(words)
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = np.zeros(len_words, 'float32')
            for d in doc:
                v[word2id[d]] += 1./self.vocab._id2tf[d]
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)
        v1 = np.dot(d1, vectors)
        v2 = np.dot(d2, vectors)
        return self.distance(v1, v2)


    def wmd_distance(self, sentence_id1, sentence_id2):
        from pyemd import emd
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = np.unique(sentence_id1 + sentence_id2)
        vectors = self.vocab.senid2vec(words)
        distance_matrix = cosine_distances(vectors)
        if np.sum(distance_matrix) == 0.0:
            return float('inf')
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = np.zeros(len_words, 'float32')
            for d in doc:
                v[word2id[d]] += 1./self.vocab._id2tf[d]
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)
        return emd(d1, d2, distance_matrix)
    

    def rwmd_distance(self, sentence_id1, sentence_id2):
        if len(sentence_id1) < 1 or len(sentence_id2) < 1:
            return float('inf')
        words = np.unique(sentence_id1 + sentence_id2)
        vectors = self.vocab.senid2vec(words)
        distance_matrix = cosine_distances(vectors)
        if np.sum(distance_matrix) == 0.0:
            return float('inf')
        len_words = len(words)

        word2id = dict(zip(words, range(len_words)))
        #print word2id
        def doc2sparse(doc):
            v = np.zeros(len_words, 'float32')
            for d in doc:
                v[word2id[d]] += 1./self.vocab._id2tf[d]
                #v[word2id[d]] += 1.
            return v/v.sum()
        d1, d2 = doc2sparse(sentence_id1), doc2sparse(sentence_id2)

        new_weights_dj = distance_matrix[:len(d1), d2>0].min(axis=1)
        new_weights_di = distance_matrix[:len(d2), d1>0].min(axis=1)

        rwmd = max(np.dot(new_weights_dj, d1), np.dot(new_weights_di, d2))
        return rwmd


    def idf_weighted_distance(self, sentence_id1, sentence_id2):
        vec1 = self.vocab.ave_vec(sentence_id1)
        vec2 = self.vocab.ave_vec(sentence_id2)
        return self.distance(vec1, vec2)
   

    def distance(self, vec1, vec2):
        return cosine(vec1, vec2)


