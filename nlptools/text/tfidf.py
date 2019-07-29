#!/usr/bin/env python3
import numpy, multiprocessing, os
from scipy.sparse import csr_matrix, lil_matrix
from functools import partial
from ..utils import zload, zdump


def n_count_ids(ids):
    '''
        calculate n_count for each id in ids

        Input:
            - ids: list of int

        Output:
            - sparse matrix with (row list, col list, data list)
    '''
    ids, doc_id = ids
    row, counts= numpy.unique(ids, return_counts=True)
    col = numpy.ones(len(row)) * doc_id

    return row, col, counts

class TFIDF:
    '''
        TF-IDF

        Input:
            - vocab: instance of .vocab, default is None
            - freqwords_path: path of freqword list, will force set the count of each word in the list to a large number
            - cached_index: path of cached index file
    '''
    def __init__(self, vocab_size = 10000, cached_index = ''):
        self.cached_index = cached_index
        self.word_idfs = None
        self.vocab_size = vocab_size
        self.load_index()

    def get_count_matrix(self, corpus_ids, corpus_len):
        '''
            count bag of words in corpus_ids

            Input:
                - corpus_ids: list or iterator of corpus ids, format of [[id, id, ...],...] 
                - corpus_len: int, length of corpus ids

            Output:
                - count matrix (scipy.sparse.csr_matrix)
        '''
        #row, col, data = [], [], []
        pool = multiprocessing.Pool(
            max(multiprocessing.cpu_count()-2, 1)
        )

        count_matrix = lil_matrix((self.vocab_size, corpus_len), dtype="int")

        for b_row, b_col, b_data in pool.imap_unordered(n_count_ids, zip(corpus_ids, range(corpus_len))):
            count_matrix[b_row, b_col] += b_data
        pool.close()
        pool.join()

        count_matrix = count_matrix.tocsr()

        return count_matrix

    def get_doc_freqs(self, cnts):
        '''
            Return word --> # of docs it appears in.
            Input:
                - count matrix
        '''
        binary = (cnts > 0).astype(int)
        freqs = numpy.array(binary.sum(1)).squeeze()
        return freqs

    def load_index(self, corpus_ids=None, corpus_len=None, retrain=False, local_use=False):
        '''
            Build or load index for corpus_ids

            Input:
                - corpus_ids: a list or an iterator of sentence_ids. Will only be used when the index is needed to train. default is None.
                - corpus_len: int, length of corpus ids. Necessary if corpus_ids is an iterator. Default is None
                - retrain: bool, check if index need to rebuild, default is False
                - local_use: bool, check if return count_matrix or word_idfs. If not, they will be saved to intern variable. Else they will be returned. Default is False

            Output: will only return when local_use=True
                - count_matrix: a sparse matrix for each id count in corpus
                - word_idfs: the idf list for a word list
        '''
        if not local_use and os.path.exists(self.cached_index) and not retrain:
            tmp = zload(self.cached_index)
            self.count_matrix = tmp[0]
            self.word_idfs = tmp[1]
            self.vocab_size = self.count_matrix.shape[0]
            return
        if corpus_ids is None:
            return
        if corpus_len is None:
            corpus_len = len(corpus_ids)
        count_matrix = self.get_count_matrix(corpus_ids, corpus_len)
        word_freqs = self.get_doc_freqs(count_matrix)
        word_idfs = numpy.log(count_matrix.shape[1] - word_freqs + 0.5) - numpy.log(word_freqs + 0.5)
        if not local_use:
            self.count_matrix = count_matrix
            self.word_idfs = word_idfs
            if self.cached_index:
                zdump((count_matrix, word_idfs), self.cached_index)
        else:
            return count_matrix, word_idfs

    def get_index(self):
        """
            return index file
        """
        return self.count_matrix, self.word_idfs

    def set_index(self, count_matrix, word_idfs):
        """
            set index 
        """
        self.count_matrix = count_matrix
        self.word_idfs = word_idfs
        self.vocab_size = self.count_matrix.shape[0]

    def search_index(self, word_ids, corpus_ids=None, topN=1, global_idfs=True):
        '''
            traditional tf-idf algorithm

            Input:
                - word_ids: token ids
                - corpus_ids: a list of token ids as corpus, if None will search from global corpus from load_index function, default is None
                - topN: int, return topN result, default is 1
                - global_idfs, bool, use global idfs calculated from large corpus when doing load_index or idfs from input corpus_ids, default is True

            Output:
                - [(score, tfidf), ...], ...
        '''
        if corpus_ids is None:
            spvec = self.text2spvec(word_ids, self.word_idfs)
            res = spvec * self.count_matrix
        else:
            count_matrix, word_idfs = self.load_index(corpus_ids, local_use=True)
            if self.word_idfs is not None and global_idfs: 
                word_idfs = self.word_idfs
            spvec = self.text2spvec(word_ids, word_idfs)
            res = spvec * count_matrix

        if len(res.data) <= topN:
            o_sort = numpy.argsort(-res.data)
        else:
            o = numpy.argpartition(-res.data, topN)[0:topN]
            o_sort = o[numpy.argsort(-res.data[o])]
        
        doc_scores = res.data[o_sort]
        doc_ids = [int(x) for x in res.indices[o_sort]]
        return list(zip(doc_ids, doc_scores))
    
    def search_index_batch(self, word_idss, topN = 1):
        """
            Process a batch of closest_docs requests multithreaded. Note: we can use plain threads here as scipy is outside of the GIL.

            Input:
                - word_idss: a list of token ids
                - topN: int, return topN result, default is 1

            Output:
                - a list of result from search_index
                
        """
        ncpus = max(multiprocessing.cpu_count() - 2, 1)
        with multiprocessing.pool.ThreadPool(ncpus) as threads:
            closest_docs = partial(self.search_index, topN=topN)
            results = threads.map(closest_docs, word_idss)
        return results
    
    def text2spvec(self, word_ids, word_idfs):
        '''
            Used in search_index
        '''
        # Count 
        wids_unique, wids_counts = numpy.unique(word_ids, return_counts=True)
        tfs = numpy.log1p(wids_counts)
        
        # Count IDF
        idfs = word_idfs[wids_unique]

        # TF-IDF
        data = numpy.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = numpy.array([0, len(wids_unique)])
        spvec = csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.vocab_size)
        )
        return spvec 

