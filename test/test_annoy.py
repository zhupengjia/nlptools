#!/usr/bin/env python
import sys
from nlptools.text.annoysearch import AnnoySearch
from nlptools.text import Tokenizer, Embedding

tok = Tokenizer(tokenizer='jieba')
emb = Embedding(w2v_word2idx='/home/pzhu/data/word2vec/zhwiki_word2idx.pkl', w2v_idx2vec='/home/pzhu/data/word2vec/zhwiki_vectors.pkl', dim=100)

s = AnnoySearch(emb, tok)
s.load_index(['天气','糟糕','气温','位置', '妇女'])
print(s.find('今天的空气很差，温度也很低'))

