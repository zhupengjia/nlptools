#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os, re, uuid, shutil, glob, traceback
from collections import defaultdict
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from .tokenizer import Tokenizer_Spacy, Tokenizer_LTP


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, nlp, regex):
        '''
            regex matcher for spacy pipeline 

            Input:
                - nlp: spacy instance
                - regex: regex dictionary, like {"key":regex}
        '''
        self.patterns = {}
        for k in regex:
            self.patterns[k] = re.compile(regex[k])
    
    def __call__(self, doc):
        for k in self.patterns:
            for match in re.finditer(self.patterns[k], doc.text):
                start, end = match.span()
                span = doc.char_span(start, end, label=k)
                doc.ents = list(doc.ents) + [span]
        return doc



class KeywordsMatcher:
    name = "keywords_matcher"
    
    def __init__(self, nlp, keywords):
        '''
            keywordsmatcher for spacy pipeline using phrasematcher

            Input:
                - nlp: spacy instance
                - keywords: keyword dictionary, like {"key":[words]}
        '''
        self.matcher = PhraseMatcher(nlp.vocab)
        patterns = {}
        for k in keywords:
            patterns[k] = [nlp(text) for text in keywords[k]]
            self.matcher.add(k, None, *patterns[k])
    
    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            doc.ents = list(doc.ents) + [span]
        return doc


class NER_Base(object):
    '''
        Parent class for other NER classes, please don't use this class directly

        Input:
            - keywords: dictionary of {entityname: keywords list file path} or {entityname: keywords list}, entity recognition via keywords list, default is None
            - ner: list, entity recognition via NER, will only remain the entity names in list. Default is None
            - regex: dictionary of {entityname: regex}, entity recognition via REGEX. Default is None
    '''
    def __init__(self, keywords = None, ner = None, regex = None):
        self.ner_names = list(keywords.keys() if keywords is not None else []) + \
                            list(regex.keys() if regex is not None else []) + \
                            list(ner if ner is not None else [])

    def _read_keywords(self, keywords):
        keywords_rebase = {}
        for k in keywords:
            if isinstance(keywords[k], list):
                keywords_rebase[k] = keywords[k]
            else:
                if not os.path.exists(keywords[k]):
                    continue
                keywords_rebase[k] = []
                with open(keywords[k]) as f:
                    for l in f:
                        l = l.strip()
                        if len(l) < 1 or l[0] == '#':
                            continue
                        l_split = [x.strip() for x in re.split(':', l, maxsplit=1)]
                        l_split = [x for x in l_split if len(x) > 0]
                        if len(l_split) < 1:
                            continue
                        entity = l_split[0].lower()
                        keywords_rebase[k].append(entity)
                        if len(l_split) == 2 and not entity in self.entity_replace:
                            self.entity_replace[entity] = l_split[1].lower()

            keywords_rebase[k] = list(set(keywords_rebase[k]))
            keywords_rebase[k].sort(key=len, reverse=True)
        return keywords_rebase


    def get(self, sentence, return_dict=False):
        '''
            get entities and marked sentence

            Input:
                - sentence: string
                - return_dict: bool, True will return like {entityname:entity,}, False will return like [(entityname:entity), ...], default is False
        '''
        entities = self.entities(sentence)
        for en, ev in entities:
            try:
                sentence = re.sub(ev, r"$"+en, sentence)
            except:
                err = traceback.format_exc()
                print(err)
        if return_dict:
            entities_dict = defaultdict(list)
            for en, ev in entities:
                entities_dict[en].append(ev) 
            entities = entities_dict
        return entities, sentence


class NER_Spacy(NER_Base, Tokenizer_Spacy):
    '''
        The NER part uses Spacy. The class inherit from NER_Base and Spacy
        
        Input:
            - keywords: dictionary of {entityname: keywords list file path} or {entityname: keywords list}, entity recognition via keywords list, default is None
            - ner: list, entity recognition via NER, will only remain the entity names in list. Default is None
            - regex: dictionary of {entityname: regex}, entity recognition via REGEX. Default is None
            - Other arguments used in text.Tokenizer_Spacy
    '''
    def __init__(self, keywords = None, ner = None, regex = None,  **args):
        NER_Base.__init__(self, keywords, ner, regex)
        if ner is not None:
            Tokenizer_Spacy.__init__(self, spacy_pipes=["ner"], **args)
        else:
            Tokenizer_Spacy.__init__(self, spacy_pipes=None, **args)
        prefix_re = re.compile(r'''^[\‘\[\{\<\(\"\']''')
        suffix_re = re.compile(r'''[\]\}\>\)\"\'\,\.\!\?]$''')
        infix_re = re.compile(r'''[\<\>\{\}\(\)\/\-~\:\,\!\'\’]''')
        from spacy.tokenizer import Tokenizer
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, \
                prefix_search=prefix_re.search, \
                suffix_search=suffix_re.search, \
                infix_finditer=infix_re.finditer)

        if regex is not None:
            regex_matcher = RegexMatcher(self.nlp, regex)
            self.nlp.add_pipe(regex_matcher, before="ner")

        if keywords is not None:
            entity_matcher = KeywordsMatcher(self.nlp, self._read_keywords(keywords))
            self.nlp.add_pipe(entity_matcher, before="ner")

    def entities(self, sentence):
        '''
            return needed entities

            Input:
                - sentence: string
        '''
        entities = []
        for ent in self.nlp(sentence).ents:
            label = ent.label_
            if label in self.ner_name_replace:
                label = self.ner_name_replace[label]
            if not label in self.ner_names:
                continue
            entities.append((label, ent.text))
        return entities


    def train(self, entities, data, spacy_model, n_iter = 50):
        '''
            Train user defined NER model via spacy api.
            
            Input:
                - entities: entity name list
                - data: format of [(text, annotations), ...]. Please check `spacy document <https://spacy.io/usage/training>`_ for more details
                - spacy_model: string, filepath to save
                - n_iter: iteration

        '''
        import random
        if 'ner' not in self.nlp.pipe_names:
            self.nlp.create_pipe('ner')
            self.nlp.add_pipe('ner')
        else:
            ner=self.nlp.get_pipe('ner')
		
        for e in entities:
            ner.add_label(e)

        other_pipes=[pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(data)
                losses={}
                for text, annotations in data:
                    self.nlp.update([text], [annotations], sgd = optimizer, losses=losses)
        self.nlp.to_disk(spacy_model)
    

class NER_LTP(NER_Base, Tokenizer_LTP):
    '''
        The NER part uses PyLTP. The class inherit from NER_Base and text.Tokenizer_LTP
        WARNING!!! need to reconstruct the code

        Input:
            - please check the needed parameters from NER_Base and text.Tokenizer_LTP
    '''
    def __init__(self, keywords = None, ner = None, regex = None, **args):
        Tokenizer_LTP.__init__(self, **args)

    
    def train_predeal(self, data):
        '''
            Predeal the training data to LTP training data format

            Input:
                - data: format of (text, [(start, end, entityname), ]), same as spacy. Please check `spacy document <https://spacy.io/usage/training>`_ for more details
        '''
        point = 0
        tokens, tags, entities = [], [], []
        for start,end,entity in data[1]:
            if start > point:
                tmpdata = data[0][point:start]
                if len(tmpdata.strip()) > 0:
                    words_ = self.seg(tmpdata, entityjoin=False)
                    tokens += words_['tokens']
                    tags += words_['tags']
                    entities += words_['entities']
            if entity in self.ner_name_replace:
                entity = self.ner_name_replace[entity]
            tmpdata = data[0][start:end]
            if len(tmpdata.strip()) < 1:
                continue
            words_ = self.seg(tmpdata, entityjoin=False)
            tokens += words_['tokens']
            tags += words_['tags']
            if len(words_['tokens']) < 2:
                entities += ['S-' + entity]
            else:
                entity_ = []
                for i in range(len(words_['tokens'])):
                    if i == 0 :
                        entity_.append('B-' + entity)
                    elif i == len(words_['tokens']) - 1:
                        entity_.append('E-' + entity)
                    else:
                        entity_.append('I-' + entity)
                entities += entity_
            point = end
        if point < len(data[0]):
            tmpdata = data[0][point:]
            if len(tmpdata.strip()) > 0:
                words_ = self.seg(tmpdata, entityjoin=False)
                tokens += words_['tokens']
                tags += words_['tags']
                entities += words_['entities']
        ltp_train = []
        for i, w in enumerate(tokens):
            ltp_train.append( '{0}/{1}#{2}'.format(w, tags[i], entities[i]) )
        return ' '.join(ltp_train)
        
    def train(self, data, maxiter=20):
        '''
            Train model via LTP

            Input:
                - data: list of tuple with format of (text, [(start, end, entityname), ])
                - maxiter: iteration number
        '''
        nfiles = len(glob.glob(self.ner_model_path))
        tmp_file = '/tmp/ner_train_' + uuid.uuid4().hex
        tmp_ner_file = tmp_file + '.ner'
        father_dir = os.path.dirname(self.ner_model_path)
        new_ner_file = os.path.join(father_dir, '{0}_trained.ner'.format(nfiles))
        with open(tmp_file, 'w') as f:
            for d in data:
                f.write(self.train_predeal(d) + '\n')
        os.system('otner learn --model {0} --reference {1} --development {2} --max-iter {3}'.format(tmp_ner_file, tmp_file, tmp_file, maxiter))
        shutil.move(tmp_ner_file, new_ner_file)
        os.remove(tmp_file)

        from pyltp import NamedEntityRecognizer
        self.ner_ins.append(NamedEntityRecognizer())
        self.ner_ins[-1].load(new_ner_file)


class NER(object):
    '''
        Entity recognition tool, integrate with several tools 

        Input:
            - tokenizer: string, choose for NER class:
                1. *spacy*: will use NER_Spacy
    '''
    def __new__(cls, tokenizer='spacy', **args):
        tokenizers = {'spacy':NER_Spacy}

        if tokenizer in tokenizers:
            return tokenizers[tokenizer](**args)
        raise('Error! No available tokenizer founded!!!')



