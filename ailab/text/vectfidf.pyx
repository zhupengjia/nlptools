#!/usr/bin/env python3
import numpy, sys, multiprocessing, time, os, pandas, scipy
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from ailab.utils import zload, zdump, setLogger


#calculate n_count for each id in ids, output a sparse matrix
def n_count_ids(ids):
    ids, doc_id = ids
    counts = Counter(ids)
    row = list(counts.keys())
    data = list(counts.values())
    col = [doc_id] * len(row)
    return row, col, data

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins):
        self.cfg = {'freqwords_path':'', 'cached_index':''}
        for k in cfg: self.cfg[k] = cfg[k]
        self.logger = setLogger(self.cfg)
        self.vocab = vocab_ins 
        self.distance_metric = 'cosine'
        self.scorelimit = 0.6
        self.freqwords = {}
        self.word_idfs = None
        self.__loadFreqwords(self.cfg['freqwords_path'])

    def __loadFreqwords(self, freqwords_path=None):
        if os.path.exists(freqwords_path):
            print('load freqword path', freqwords_path)
            with open(freqwords_path) as f:
                for w in f.readlines():
                    for i in self.vocab.sentence2id(w.strip(), ngrams=1, useBE=False, addforce=True):
                        self.freqwords[i] = 0

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

    #Return word --> # of docs it appears in.
    def get_doc_freqs(self, cnts):
        binary = (cnts > 0).astype(int)
        freqs = numpy.array(binary.sum(1)).squeeze()
        return freqs

    def load_index(self, corpus_ids=None, retrain=False, local_use=False):
        if not local_use and os.path.exists(self.cfg['cached_index']) and not retrain:
            tmp = zload(self.cfg['cached_index'])
            self.count_matrix = tmp[0]
            self.word_idfs = tmp[1]
            return
        count_matrix = self.get_count_matrix(corpus_ids)
        word_freqs = self.get_doc_freqs(count_matrix)
        word_idfs = numpy.log(count_matrix.shape[1] - word_freqs + 0.5) - numpy.log(word_freqs + 0.5)
        if not local_use:
            self.count_matrix = count_matrix
            self.word_idfs = word_idfs
            if len(self.cfg['cached_index']) > 0:
                zdump((count_matrix, word_idfs), self.cfg['cached_index'])
                self.vocab.save()
        else:
            return count_matrix, word_idfs


    def tfidf(self, word_ids, sentence_ids, word_idfs):
        tf = self.tf(word_ids, sentence_ids)
        idf = word_idfs[word_ids]
        self.logger.debug('VecTFIDF: word_ids, ' + str(word_ids) + ' sentence_ids' + str(sentence_ids) + ' tf,' + str(tf) + " idf," + str(idf))
        return tf*idf

    #vec TF-IDF
    def search(self, word_ids, corpus_ids, topN=1, global_idfs=True):
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

    
    #traditional TF-IDF algorithms
    def search_index(self, word_ids, corpus_ids=None, topN=1, global_idfs=True):
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
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        ncpus = max(multiprocessing.cpu_count() - 2, 1)
        with multiprocessing.pool.ThreadPool(ncpus) as threads:
            closest_docs = partial(self.search_index, topN=topN)
            results = threads.map(closest_docs, word_idss)
        return results
    
    def text2spvec(self, word_ids, word_idfs):
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



