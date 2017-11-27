#!/usr/bin/env python3
import re, sys, time, json
from ailab.utils import zload, zdump, normalize
from ailab.text import *
from ailab.text.vectfidf import *
from ailab.zoo.comprehension.DrQA.model import DocReader

cfg = {'APPNAME':'test', 'vec_len':300, 'w2v_word2idx':'/home/pzhu/data/word2vec/en/word2idx_2000000.pp', 'w2v_idx2vec':'/home/pzhu/data/word2vec/en/weight_2000000.npy',  'LANGUAGE':'en', 'cached_w2v':'data/w2v.pkl', 'cached_vocab':'data/vocab.pkl', 'cached_index':'data/tfidf.index', 'freqwords_path':'data/en_freqwords.txt', 'vocab_size':18}
e = Embedding(cfg)
s = Segment(cfg)
v = Vocab(cfg, s, e, 3, hashgenerate=False)
#v2 = Vocab(cfg, s, e, 1)
t = VecTFIDF(cfg, v)

reader_params = DocReader.load('/home/pzhu/data/read_comprehension/SQuAD/DrQA/reader/single.mdl')
#for w in reader_params['word_dict']:
#    v[w] = reader_params['word_dict'][w]
#
#corpus, corpus_ids = [], []
#with open('data/accenture_policy.txt') as f:
#    for l in f:
#        text = json.loads(l)['text']
#        text = [x for x in re.split('[\n\.\[\]]', text) if len(x)>2]
#        text_ids = [v.sentence2id(x) for x in text]
#        corpus += text
#        corpus_ids += text_ids
#zdump((corpus, corpus_ids), 'data/accenture_policy.pkl')
#v.save()

corpus, corpus_ids = zload('data/accenture_policy.pkl')
#
t.load_index(corpus_ids)
#
query = 'dress code'
query_id = v.sentence2id(query)

ranked_doc = t.search_index(query_id,5)
ranked_doc_indexes = list(zip(*ranked_doc))[0]
ranked_doc = [corpus[i] for i in ranked_doc_indexes]
ranked_doc_ids = [corpus_ids[i] for i in ranked_doc_indexes]
print(ranked_doc, ranked_doc_ids)
print(ranked_doc_indexes)

for k in reader_params['args'].__dict__:
    cfg[k] = reader_params['args'].__dict__[k]

reader = DocReader(cfg, v, reader_params['feature_dict'], reader_params['state_dict'])


#print(reader_params['args'].parse_args())

#print(reader['pitch'])
#print(v[normalize('pitch')])
#print(reader)

