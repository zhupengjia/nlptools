#!/usr/bin/env python
import os, re
from nlptools.text.ner import NER_Spacy



config = {
    #"ner":["PERSON"], 
    "keywords":{
        "CUISINE": "/home/pzhu/work/chatbot/chatbot-end2end/data/babi/entities/cuisine.txt",
        "ANIMAL": ['tree kangaroo', 'giant sea spider', "cat", "dog"]
        },
    "regex":{ 
        "REST_INFO": 'resto_\w*'}
    }

ner = NER_Spacy(**config) 

sentence = "resto_rome_chep. Instead could it be with british food? This is a text about Barack Obama and a tree kangaroo."

print(ner.nlp.pipe_names)

print(ner.get(sentence))
