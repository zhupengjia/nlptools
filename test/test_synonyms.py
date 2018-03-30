#!/usr/bin/env python

import time
from ailab.text.synonyms import Synonyms
from ailab.text import Embedding, Segment

cfg = {'synonyms_path': '/home/pzhu/data/word2vec/en/synonyms_en.pkl',\
        'tokenizer': 'en',\
        'w2v_word2idx': '/home/pzhu/data/word2vec/en/word2idx_2000000.pp',\
        'w2v_idx2vec': '/home/pzhu/data/word2vec/en/weight_2000000.npy'}
emb = Embedding(cfg)

s = Synonyms(cfg, emb)
t1 = time.time()
synonyms = s('chinese', 10, 0.5)
t2 = time.time()
print(synonyms, t2-t1)

