#!/usr/bin/env python
import sys
from nlptools.text.tokenizer import Tokenizer_Simple, Tokenizer_Spacy
from nlptools.text.bpe import BytePair


#with open('corpus.en') as f:
#    corpus = f.read()
corpus = 'Dolphins are a widely distributed and diverse group of aquatic mammals. They are an informal grouping within the order Cetacea, excluding whales and porpoises, so to zoologists the grouping is paraphyletic'

t = Tokenizer_Simple()
v = BytePair(vocab_size = 100000, code_file='/home/pzhu/work/sentence_embedding/SNLI/data/bpecodes')
tokens = t(corpus)

v(tokens)

#v.learn(bpe_size = 1000)

wordids = v.words2id(tokens)
recovered = v.id2words(wordids)
print(' '.join(recovered))



