#!/usr/bin/env python3
import re, json
from ailab.zoo.comprehension.DrQA.drqa import DrQA

cfg = {'APPNAME':'test', 'vec_len':300, 'w2v_word2idx':'/home/pzhu/data/word2vec/en/word2idx_2000000.pp', 'w2v_idx2vec':'/home/pzhu/data/word2vec/en/weight_2000000.npy',  'LANGUAGE':'en', 'cached_w2v':'data/w2v.pkl', 'cached_vocab':'data/vocab.pkl', 'cached_index':'data/tfidf.index', 'freqwords_path':'data/en_freqwords.txt', 'vocab_size':18, 'drqa_reader_path':'/home/pzhu/data/read_comprehension/SQuAD/DrQA/reader/single.mdl', 'drqa_data_cache':'data/accenture_policy.pkl'}

d = DrQA(cfg)
d.load_reader()
with open('data/accenture_policy.txt') as f:
    documents = re.split('\n',f.read())
documents = [json.loads(l)['text'] for l in documents if len(l.strip())>0]
d.train(documents)

query = 'dress code'
d.search(query)

