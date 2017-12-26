#!/usr/bin/env python2
# -*- coding:utf-8 -*-

import logging
from gensim import interfaces, utils
from six import string_types
from gensim.corpora.dictionary import Dictionary

class Corpus(interfaces.CorpusABC):
    def __init__(self, docbow):
        super(Corpus, self).__init__()
        self.docbow = docbow

    def __iter__(self):
        for db in self.docbow:
            yield db


