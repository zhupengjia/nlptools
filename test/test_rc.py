#!/usr/bin/env python3
import re, sys, time, json
from ailab.utils import zload, zdump
from ailab.text import *
from ailab.text.vectfidf import *

cfg = {'APPNAME':'test', 'vec_len':300, 'w2v_word2idx':'/home/pzhu/data/word2vec/en/word2idx_2000000.pp', 'w2v_idx2vec':'/home/pzhu/data/word2vec/en/weight_2000000.npy',  'LANGUAGE':'en', 'cached_w2v':'/tmp/w2v.pkl', 'cached_vocab':'/tmp/vocab.pkl', 'cached_index':'data/tfidf.index', 'freqwords_path':'data/en_freqwords.txt'}
e = Embedding(cfg)
s = Segment(cfg)
v = Vocab(cfg, s, e, 3)
t = VecTFIDF(cfg, v)

#corpus, corpus_ids = [], []
#with open('data/accenture_policy.txt') as f:
#    for l in f:
#        text = json.loads(l)['text']
#        text = [x for x in re.split('[\n\.\[\]]', text) if len(x)>2]
#        text_ids = [v.sentence2id(x) for x in text]
#        corpus += text
#        corpus_ids += text_ids
#zdump((corpus, corpus_ids), 'data/accenture_policy.pkl')

corpus, corpus_ids = zload('data/accenture_policy.pkl')

t1 = time.time()
t.load_index(corpus_ids)
t2 = time.time()
print(t2-t1)



