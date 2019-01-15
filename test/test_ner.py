#!/usr/bin/env python
import os, re
from spacy.util import get_lang_class, get_data_path
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = get_lang_class("en")()

config = {
    "ner":[], 
    "keywords":{
        "CUISINE": ["british", "cantonese"],
        "ANIMAL": ['tree kangaroo', 'giant sea spider', "cat", "dog"]
        },
    "regex":{ 
        "REST_INFO": 'resto_\w*'}
    }

sentence = "resto_rome_chep. Instead could it be with british food? This is a text about Barack Obama and a tree kangaroo."

class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, nlp):
        self.patterns = {}
        for k in config["regex"]:
            self.patterns[k] = re.compile(config["regex"][k])
    
    def __call__(self, doc):
        for k in self.patterns:
            for match in re.finditer(self.patterns[k], doc.text):
                start, end = match.span()
                span = doc.char_span(start, end, label=k)
                doc.ents = list(doc.ents) + [span]
        return doc



class KeywordsMatcher:
    name = "keywords_matcher"
    
    def __init__(self, nlp):
        self.matcher = PhraseMatcher(nlp.vocab)
        patterns = {}
        for k in config["keywords"]:
            patterns[k] = [nlp(text) for text in config["keywords"][k]]
            self.matcher.add(k, None, *patterns[k])
    
    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            doc.ents = list(doc.ents) + [span]
        return doc


for name in ["ner"]:
    component = nlp.create_pipe(name)
    nlp.add_pipe(component)

data_path = os.path.join(get_data_path(), "en/en_core_web_sm-2.0.0")
nlp.from_disk(data_path)

entity_matcher = KeywordsMatcher(nlp)
regex_matcher = RegexMatcher(nlp)
nlp.add_pipe(regex_matcher, after="ner")
nlp.add_pipe(entity_matcher, after="regex_matcher")


for ent in nlp(sentence).ents:
    print(ent.label_, ent.text)

