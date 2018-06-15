#!/usr/bin/env python3
from nlptools.text.tokenizer import Segment_CoreNLP

cfg = {'CORENLP_URL':'http://127.0.0.1:9000'}
t = Segment_CoreNLP(cfg)

text = 'Change syntax themes, default project pages, and more in preferences.\n hello world'

print(t.seg_sentence(text))


