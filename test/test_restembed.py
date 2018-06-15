#!/usr/bin/env python3
from nlptools.text.embedding import Embedding_Rest

cfg = {'embedding_restapi':'http://127.0.0.1:8000/api/query/' }
t = Embedding_Rest(cfg)

#text = 'Change syntax themes, default project pages, and more in preferences.\n hello world'
text = "今天"

print(t[text])



