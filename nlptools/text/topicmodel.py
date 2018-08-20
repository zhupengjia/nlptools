#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

import os, gensim
import pandas as pd
import numpy as np
from gensim import interfaces, models, similarities
from ..utils import zload, zdump

class Corpus(interfaces.CorpusABC):
    '''
        Corpus class inherit from gensim.interfaces.CorpusABC
        
        Input:
            - docbow: format of [(id, tf), ...], can be the output from text.vocab.doc2bow
                
    '''
    def __init__(self, docbow):
        super(Corpus, self).__init__()
        self.docbow = docbow

    def __iter__(self):
        for i, db in enumerate(self.docbow):
            yield db

    def __len__(self):
        return len(self.docbow)


class TopicModel(object):
    '''
        Parent class for LDA and LSI, please don't use this class directly
    '''
    def __init__(self, N_topic):
        self.N_topic = N_topic 

    def __getitem__(self, corpus):
        return np.transpose(gensim.matutils.corpus2dense(self.model[corpus], \
                num_terms=self.N_topic))


class LSI(TopicModel):
    '''
        LSI

        Input:
            - lsi_path: saved path for lsi model
            - N_topic: topic number
    '''
    def __init__(self, lsi_path, N_topic):
        TopicModel.__init__(self, N_topic)
        self.lsi_path = lsi_path

    def build(self, corpus):
        if not os.path.exists(self.lsi_path):
            self.model= models.LsiModel(corpus,
                                   num_topics=self.N_topic,
                                   onepass=False)
            self.model.save(self.lsi_path)
        else:
            self.model = models.LsiModel.load(self.lsi_path)
   

class LDA(TopicModel):
    '''
        LDA

        Input:
            - lda_path: saved path for lda model
            - N_topic: topic number
    '''
    def __init__(self, lda_path, N_topic):
        TopicModel.__init__(self, N_topic)
        self.lda_path = lda_path

    def build(self, corpus):
        if not os.path.exists(self.lda_path):
            self.model = models.LdaMulticore(corpus,
                                       num_topics=self.N_topic)
            self.model.save(self.lda_path)
        else:
            self.model = models.LdaMulticore.load(self.lda_path)



