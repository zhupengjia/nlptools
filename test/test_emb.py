#!/usr/bin/env python3
import sys
from nlptools.text.embedding import Embedding

#cfg = {'dynamodb':'word_vectors_eng', 'vec_len':300, 'RETURNBASE64':1}
cfg = {'embedding_restapi':'http://127.0.0.1:8000/api/query/', 'vec_len':100}
e = Embedding(cfg)

print(sys.argv[1], e[sys.argv[1]])

