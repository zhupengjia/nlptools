#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
from ..utils import zload, zdump

class TopicModel(object):
    def __init__(self, cfg, question_db_tokens):
        self.question_db = question_db_tokens
        self.dictionary = corpora.Dictionary(self.question_db)
        self.question = [self.dictionary.doc2bow(text) for text in self.question_db]

        tf_idf = models.TfidfModel(self.question)
        self.corpus_tf_idf = tf_idf[self.question]
        if cfg["N_topic"]<1:
            self.num_of_topics = len(self.question_db)*cfg["N_topic"]
        else:
            self.num_of_topics = cfg["N_topic"]

        self.cfg = cfg
        #self.build_lda()
        self.build_lsi()

class LSI(TopicModel):
    def __init__(self, cfg, question_db_tokens):
        TopicModel.__init__(self, cfg, question_db_tokens)

    def build(self):
        if not os.path.exists(self.cfg["lsi_path"]):
            self.lsi = models.LsiModel(self.corpus_tf_idf,
                                   id2word=self.dictionary,
                                   num_topics=self.num_of_topics,
                                   onepass=False)
            self.lsi_index = similarities.MatrixSimilarity(self.lsi[self.question])
            zdump((self.lsi, self.lsi_index), self.cfg["lsi_path"])
        else:
            self.lsi, self.lsi_index = zload(self.cfg["lsi_path"])
    
    def query(self, question):
        query_bow = self.dictionary.doc2bow(question)
        query_lsi = self.lsi[query_bow]
        scores = self.lsi_index[query_lsi]
        return scores

class LDA(TopicModel):
    def __init__(self, cfg, question_db_tokens):
        TopicModel.__init__(self, cfg, question_db_tokens)

    def build(self):
        if not os.path.exists(self.cfg["lda_path"]):
            self.lda = models.LdaMulticore(self.corpus_tf_idf,
                                       id2word=self.dictionary,
                                       num_topics=self.num_of_topics,
                                       passes=5)
            self.lda_index = similarities.MatrixSimilarity(self.lda[self.question])
            zdump((self.lda, self.lda_index), self.cfg["lda_path"])
        else:
            self.lda, self.lda_index = zload(self.cfg["lda_path"])

    def query(self, question):
        query_bow = self.dictionary.doc2bow(question)
        query_lda = self.lda[query_bow]
        scores = self.lda_index[query_lda]
        return scores
