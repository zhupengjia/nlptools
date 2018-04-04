#!/usr/bin/env python

import time, sys
from ailab.text import Synonyms, Embedding

cfg = {'synonyms_path': '/home/pzhu/data/word2vec/en/synonyms_en.pkl',\
        'tokenizer': 'en',\
        'w2v_word2idx': '/home/pzhu/data/word2vec/en/word2idx_2000000.pp',\
        'w2v_idx2vec': '/home/pzhu/data/word2vec/en/weight_2000000.npy'}
emb = Embedding(cfg)

s = Synonyms(cfg, emb)
t1 = time.time()
synonyms = s(sys.argv[1], 100, 0.5)
t2 = time.time()
print(t2-t1)
for i in range(len(synonyms[0])):
    print(synonyms[0][i],'\t\t', synonyms[1][i])


