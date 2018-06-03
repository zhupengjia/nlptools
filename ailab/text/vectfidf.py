#!/usr/bin/env python3
import numpy, sys, multiprocessing, time, os, pandas, scipy
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from ..utils import zload, zdump, setLogger
from .vocab import Vocab

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

def n_count_ids(ids):
    '''
        calculate n_count for each id in ids
        
        Input: 
            - ids: list of int

        Output:
            - sparse matrix with (row list, col list, data list)
    '''
    ids, doc_id = ids
    counts = Counter(ids)
    row = list(counts.keys())
    data = list(counts.values())
    col = [doc_id] * len(row)
    return row, col, data

class VecTFIDF(object):
    '''
        Modified TF-IDF algorithm with wordvector

        Input:
            - vocab: instance of text.vocab, default is None
            - freqwords_path: path of freqword list, will force set the count of each word in the list to a large number
            - cached_index: path of cached index file
    '''
    def __init__(self, vocab = None, cached_index = '', freqwords_path=''):
        self.cached_index = cached_index
        self.freqwords_path = freqwords_path
        self.vocab = vocab 
        if self.vocab is None: self.vocab = Vocab()
        self.distance_metric = 'cosine'
        self.scorelimit = 0.6
        self.freqwords = {}
        self.word_idfs = None
        self.__loadFreqwords(self.freqwords_path)

    def __loadFreqwords(self, freqwords_path=None):
        if os.path.exists(freqwords_path):
            print('load freqword path', freqwords_path)
            with open(freqwords_path) as f:
                for w in f.readlines():
                    for i in self.vocab.sentence2id(w.strip(), ngrams=1, useBE=False, update=True):
                        self.freqwords[i] = 0


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
        sentence_vec = self.vocab.senid2vec(sentence_ids)
        word_vec = self.vocab.senid2vec(word_ids)
        score = 1/(1.+pairwise_distances(word_vec, sentence_vec, metric = self.distance_metric))
        score[score < self.scorelimit] = 0
        return score.sum(axis = 1)


    def tf(self, word_ids, sentence_ids):
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
   

    def get_count_matrix(self, corpus_ids):
        '''
            count bag of words in corpus_ids

            Input:
                - corpus_ids: corpus ids, format of [[id, id, ...],...] 

            Output:
                - count matrix (scipy.sparse.csr_matrix)
        '''
        row, col, data = [], [], []
        pool = multiprocessing.Pool(
            max(multiprocessing.cpu_count()-2, 1)
        )

        for b_row, b_col, b_data in pool.imap_unordered(n_count_ids, zip(corpus_ids, range(len(corpus_ids)))):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
        pool.close()
        pool.join()
        
        count_matrix = scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(self.vocab.vocab_size, len(corpus_ids))
        )
        count_matrix.sum_duplicates()
         
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


    def load_index(self, corpus_ids=None, retrain=False, local_use=False):
        '''
            Build or load index for corpus_ids

            Input:
                - corpus_ids: a list of sentence_ids. Will only be used when the index is needed to train. default is None.
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
            return
        count_matrix = self.get_count_matrix(corpus_ids)
        word_freqs = self.get_doc_freqs(count_matrix)
        word_idfs = numpy.log(count_matrix.shape[1] - word_freqs + 0.5) - numpy.log(word_freqs + 0.5)
        if not local_use:
            self.count_matrix = count_matrix
            self.word_idfs = word_idfs
            if len(self.cached_index) > 0:
                zdump((count_matrix, word_idfs), self.cached_index)
                self.vocab.save()
        else:
            return count_matrix, word_idfs


    def tfidf(self, word_ids, sentence_ids, word_idfs):
        '''
            calculate tfidf for each id in a token list

            Input:
                - word_ids: token ids
                - sentence_ids: token ids in a sentence
                - word_idfs: idf for token ids

            Output:
                tfidf list
        '''
        tf = self.tf(word_ids, sentence_ids)
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
        tfidf = corpus_ids.apply(lambda x: self.tfidf(word_ids, x, word_idfs).sum()).as_matrix()
        
        scores = numpy.argsort(tfidf)[::-1]
        scores = [int(s) for s in scores[:topN]]
        tfidf = tfidf[scores]
        return list(zip(scores, tfidf))

    
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
        spvec = scipy.sparse.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.vocab.vocab_size)
        )
        return spvec 



