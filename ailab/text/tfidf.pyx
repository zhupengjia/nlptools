#!/usr/bin/env python3
import numpy, sys, multiprocessing, time, os, pandas, scipy
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from ailab.utils import zload, zdump, setLogger


#calculate n_count for each id in ids, output a sparse matrix
def n_count(ids):
    ids, doc_id = ids
    counts = Counter(ids)
    row = list(counts.keys())
    data = list(counts.values())
    col = [doc_id] * len(row)
    return row, col, data


class TFIDF(object):
    def __init__(self, cfg, vocab_ins=None):
        self.cfg = cfg
        self.logger = setLogger(self.cfg)
        self.vocab = vocab_ins 

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


    def load_index(self, corpus_ids=None, retrain=False):
        if 'tfidf_index' in self.cfg and os.path.exists(self.cfg['tfidf_index']) and not retrain:
            self.index = zload(self.cfg['tfidf_index'])
            return
          
        count_matrix = self.get_count_matrix(corpus_ids)
        tfidf = self.get_tfidf_matrix(count_matrix)
        freqs = self.get_doc_freqs(count_matrix)
        self.index = {'count_matrix': count_matrix, 'tfidf': tfidf, 'freqs': freqs}
        zdump(self.index, self.cfg['tfidf_index'])

    def search(self, word_ids, topN=1):
        spvec = self.text2spvec(word_ids)
         
        res = spvec * self.index['count_matrix']
        print(res)
        print(res.data)
        if len(res.data) <= topN:
            o_sort = numpy.argsort(-res.data)
        else:
            o = numpy.argpartition(-res.data, topN)[0:topN]
            o_sort = o[numpy.argsort(-res.data[o])]
        
        doc_scores = res.data[o_sort]
        print('o_sort', o_sort, doc_scores)

    
    
    def text2spvec(self, word_ids):
        # Count 
        wids_unique, wids_counts = numpy.unique(word_ids, return_counts=True)
        tfs = numpy.log1p(wids_counts)
        
        # Count IDF
        Ns = self.index['freqs'][wids_unique]
        idfs = numpy.log((self.index['count_matrix'].shape[1] - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = numpy.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = numpy.array([0, len(wids_unique)])
        spvec = scipy.sparse.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.vocab.vocab_hash_size)
        )
        return spvec 


