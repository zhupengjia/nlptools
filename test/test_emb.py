#!/usr/bin/env python3
from ailab.text.embedding import Embedding

cfg = {'dynamodb':'word_vectors_eng', 'vec_len':300, 'RETURNBASE64':1}
e = Embedding(cfg)

print(e['hellosdgsadgsdf'])

