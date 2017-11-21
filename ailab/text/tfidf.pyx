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



