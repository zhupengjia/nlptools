#!/usr/bin/env python3
import re, sys
from ailab.text import *
from ailab.text.tfidf import *

cfg = {'APPNAME':'test', 'vec_len':10, 'LANGUAGE':'en', 'cached_w2v':'/tmp/w2v.pkl', 'cached_vocab':'/tmp/vocab.pkl', 'cached_index':'/tmp/tfidf.index', 'freqwords_path':'data/en_freqwords.txt'}
#e = Embedding(cfg)
s = Segment(cfg)
v = Vocab(cfg, s, None, 3)
t = TFIDF(cfg, v)
corpus = '''In information retrieval, tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.[1] It is often used as a weighting factor in information retrieval, text mining, and user modeling. The tf-idf value increases proportionally to the number of times a word appears in the document, but is often offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. Nowadays, tf-idf is one of the most popular term-weighting schemes. For instance, 83% of text-based recommender systems in the domain of digital libraries use tf-idf.[2]
Variations of the tf–idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. tf–idf can be successfully used for stop-words filtering in various subject fields including text summarization and classification.
One of the simplest ranking functions is computed by summing the tf–idf for each query term; many more sophisticated ranking functions are variants of this simple model.Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.'''

corpus = [x for x in re.split('[\n\.\[\]]', corpus) if len(x)>2]

v.doc2ids(corpus)

sys.exit()

corpus_ids = [v.sentence2id(x) for x in corpus]
#print('-'*60)
#for i in corpus_ids:
#    print(i)
#print('-'*60)

count_matrix = t.get_count_matrix(corpus_ids)
tfidf = t.get_tfidf_matrix(count_matrix)
freqs = t.get_doc_freqs(count_matrix)
print(tfidf)


