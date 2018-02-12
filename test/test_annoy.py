#!/usr/bin/env python
from nlptools.text.annoysearch import AnnoySearch
from nlptools.text import Segment, Embedding

cfg = {'TOKENIZER':'jieba', 'w2v_word2idx':'/home/pzhu/data/word2vec/zhwiki_word2idx.pkl', 'w2v_idx2vec':'/home/pzhu/data/word2vec/zhwiki_vectors.pkl', 'vec_len':100}

emb = Embedding(cfg)
s = AnnoySearch(cfg, emb)
s.load_index(['天气','糟糕','气温','位置', '妇女'])
print(s.find('今天的空气很差，温度也很低'))

