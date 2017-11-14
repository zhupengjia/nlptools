#!/usr/bin/env python3
import numpy, sys, multiprocessing, time, os, pandas, scipy
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from ailab.utils import zload, zdump, n_count

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins=None):
        self.cfg = cfg
        self.vocab = vocab_ins 
        self.n_containing = numpy.vectorize(self.n_containing_id)
        self.distance_metric = 'cosine'
        self.scorelimit = 0.6


    def n_sim(self, word_ids, sentence_ids):
        if len(sentence_ids) < 1:
            return 0
        if isinstance(word_ids, int):
            word_ids = [word_ids]
        sentence_vec = self.vocab.senid2vec(sentence_ids)
        word_vec = self.vocab.senid2vec(word_ids)
        score = 1/(1.+pairwise_distances(word_vec, sentence_vec, metric = self.distance_metric))
        score[score < self.scorelimit] = 0
        return score.sum(axis = 1)


    #calculate tf
    def tf(self, word_ids, sentence_ids):
        if len(sentence_ids) == 0:
            return 0
        return self.n_sim(word_ids, sentence_ids)/len(sentence_ids)
   

    #count bag of words in corpus_ids
    def get_count_matrix(self, corpus_ids):
        row, col, data = [], [], []
        pool = multiprocessing.Pool(
            multiprocessing.cpu_count()-2
        )

        for b_row, b_col, b_data in pool.imap_unordered(n_count, zip(corpus_ids, range(len(corpus_ids)))):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
        pool.close()
        pool.join()
        
        count_matrix = scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(self.vocab.vocab_hash_size, len(corpus_ids))
        )
        count_matrix.sum_duplicates()

        return count_matrix


    # """Convert the word count matrix into tfidf one.
    # tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    # * tf = term frequency in document
    # * N = number of documents
    # * Nt = number of occurences of term in all documents
    # """
    def get_tfidf_matrix(self, cnts):
        Ns = self.get_doc_freqs(cnts)
        idfs = numpy.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        idfs = scipy.sparse.diags(idfs, 0)
        tfs = cnts.log1p()
        tfidfs = idfs.dot(tfs)
        return tfidfs, Ns


    #Return word --> # of docs it appears in.
    def get_doc_freqs(self, cnts):
        binary = (cnts > 0).astype(int)
        freqs = numpy.array(binary.sum(1)).squeeze()
        return freqs


    #build index for each word in large corpus, for idf
    def load_index(self, corpus_ids=None, retrain=False):
        if 'cached_index' in self.cfg and os.path.exists(self.cfg['cached_index']):
            tmp = zload(self.cfg['cached_index'])
            self.index_word2doc = tmp[0]
            self.len_corpus = tmp[1]
            if len(tmp) > 2:
                self.corpus_lens = tmp[2]
            return
        
        #count_matrix = self.get_count_matrix(corpus_ids)
        #tfidf, freqs = self.get_tfidf_matrix(count_matrix)

        self.index_word2doc = {}
        self.len_corpus = len(corpus_ids)
        self.corpus_lens = numpy.array([len(x) for x in corpus_ids])
        all_ids = list(set([item for sublist in corpus_ids for item in sublist]))
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)

        for i in all_ids:
            t1 = time.time()
            func_n_count = partial(n_count, i)
            ncounts = numpy.array(pool.map(func_n_count, corpus_ids), 'int32')
            self.index_word2doc[i] = {j:ncounts[j] for j in ncounts.nonzero()[0]}
            del func_n_count
            t2 = time.time()
            print('building index, ', i, len(all_ids), t2-t1)

        if 'cached_index' in self.cfg:
            zdump((self.index_word2doc, self.len_corpus, self.corpus_lens), self.cfg['cached_index'])


    def n_containing_id(self, word_id):
        if not word_id in self.index_word2doc:
            return 0
        return len(self.index_word2doc[word_id])

    def idf(self, word_ids):
        #return numpy.log(self.len_corpus) - numpy.log(1. + self.n_containing(word_ids))
        Ns = self.n_containing(word_ids)
        return numpy.log(self.len_corpus - Ns + 0.5) - numpy.log(Ns + 0.5)

    def tfidf(self, word_ids, sentence_ids):
        return self.tf(word_ids, sentence_ids) * self.idf(word_ids)

    #vec TF-IDF
    def search(self, word_ids, corpus_ids, topN=1):
        corpus_ids = pandas.Series(corpus_ids)
        tfidf = corpus_ids.apply(lambda x: self.tfidf(word_ids, x).sum()).as_matrix()
        scores = numpy.argsort(tfidf)[::-1]
        return list(zip(scores[:topN], tfidf[scores]))

    #traditional TF-IDF algorithms
    def search_by_index(self, word_ids):
        idf = self.idf(word_ids)
        tfidf = {}
        for i,wid in enumerate(word_ids):
            if not wid in self.index_word2doc:
                continue
            tf = self.index_word2doc[wid]
            for k in tf.keys():
                tf[k] = tf[k]/self.corpus_lens[k]
                if not k in tfidf:
                    tfidf[k] = tf[k]*idf[i]
                else:
                    tfidf[k] += tf[k]*idf[i]
        
        scores = sorted(tfidf.items(), key=lambda x:x[1], reverse=True)
        return scores

