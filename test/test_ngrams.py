#!/usr/bin/env python3
import re, sys
from nlptools.text import Embedding, Segment, Vocab

cfg = {'vec_len':10, 'LANGUAGE':'en', 'cached_w2v':'/tmp/w2v.pkl', 'cached_vocab':'/tmp/vocab.pkl', 'cached_index':'/tmp/tfidf.index'}
e = Embedding(cfg)
s = Segment(cfg)
v = Vocab(cfg, s, e, 3)

v.sentence2id('hello world, hi members')


