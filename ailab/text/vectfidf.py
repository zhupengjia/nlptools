#!/usr/bin/env python3
import numpy, scipy
from multiprocessing import Pool as ProcessPool

class VecTFIDF(object):
    def __init__(self, cfg, vocab_ins = None):
        self.cfg = cfg
        self.vocab = []
        self.vocab = vocab_ins
       

    def ngrams(self, N):
        pass

    def load_index(self, corpus_ids=None, retrain=False):
        
        pass



