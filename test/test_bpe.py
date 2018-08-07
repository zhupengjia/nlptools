#!/usr/bin/env python
import sys
from nlptools.text.tokenizer import Tokenizer_Simple
from nlptools.text.bpe import BytePair


with open('corpus.en') as f:
    corpus = f.read()

t = Tokenizer_Simple()
v = BytePair(vocab_size = 100000, code_file='bpe.ref')
tokens = t(corpus)

v(tokens)

v.learn(bpe_size = 1000)

wordids = v.words2id(tokens)
recovered = v.id2words(wordids)
print(' '.join(recovered))



