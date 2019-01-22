#!/usr/bin/env python
from nlptools.text.sentence_embedding import Sentence_Embedding

se = Sentence_Embedding(bert_model_name="/home/pzhu/data/bert/bert-base-uncased", device="cuda:0")

sentences = ["Economy is hitting more turbulence, but the shutdown leaves investors partly in the dark","Why investors are starting to pay attention to the government shutdown"]

for embedding in se(sentences, batch_size=100):
    print(embedding.shape)




