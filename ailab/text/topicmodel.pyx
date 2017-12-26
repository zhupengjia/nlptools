#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, gensim
import pandas as pd
import numpy as np
from gensim import interfaces, models, similarities
from ..utils import zload, zdump

class Corpus(interfaces.CorpusABC):
    def __init__(self, docbow):
        super(Corpus, self).__init__()
        self.docbow = docbow

    def __iter__(self):
        for i, db in enumerate(self.docbow):
            yield db

    def __len__(self):
        return len(self.docbow)


class TopicModel(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __getitem__(self, corpus):
        return np.transpose(gensim.matutils.corpus2dense(self.model[corpus], \
                num_terms=self.cfg['N_topic']))


class LSI(TopicModel):
    def __init__(self, cfg):
        TopicModel.__init__(self, cfg)

    def build(self, corpus):
        if not os.path.exists(self.cfg["lsi_path"]):
            self.model= models.LsiModel(corpus,
                                   num_topics=self.cfg['N_topic'],
                                   onepass=False)
            self.model.save(self.cfg['lsi_path'])
        else:
            self.model = models.LsiModel.load(self.cfg['lsi_path'])
   

class LDA(TopicModel):
    def __init__(self, cfg):
        TopicModel.__init__(self, cfg)

    def build(self, corpus):
        if not os.path.exists(self.cfg["lda_path"]):
            self.model = models.LdaMulticore(corpus,
                                       num_topics=self.cfg['N_topic'])
            self.model.save(self.cfg['lda_path'])
        else:
            self.model = models.LdaMulticore.load(self.cfg['lda_path'])



