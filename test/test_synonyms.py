#!/usr/bin/env python

import time, sys
from nlptools.text import Synonyms, Embedding

cfg = {'synonyms_path': '/home/pzhu/data/word2vec/synonyms_zhwiki.pkl',\
        'tokenizer': 'jieba',\
        'vec_len': 100,\
        'w2v_word2idx': '/home/pzhu/data/word2vec/zhwiki_word2idx.pkl',\
        'w2v_idx2vec': '/home/pzhu/data/word2vec/zhwiki_vectors.pkl'}
emb = Embedding(cfg)

s = Synonyms(cfg, emb)
while True:
    text = input(":")
    synonyms = s(text, 100, 0.5)
    for i in range(len(synonyms[0])):
        print(synonyms[0][i],'\t\t', synonyms[1][i])


