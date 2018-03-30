#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os, numpy, re, uuid, shutil, glob, sys
from .tokenizer import *


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class NER_Base(object):
    '''
        Parent class for other NER classes, please don't use this class directly

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - keywords: dictionary of {entityname: keywords list file path}, entity recognition via keywords list
                    - ner: list, entity recognition via NER, will only remain the entity names in list
                    - regex: dictionary of {entityname: regex}, entity recognition via REGEX.
                    - stopwords_path: the path of stopwords
                    - ner_name_replace: dictionary, replace the entity name to the mapped name
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.custom_regex = {}
        self.keywords_regex = {}
        self.keywords_index = None
        self.entity_replace = {}
        if 'keywords' not in self.cfg or self.cfg['keywords'] is None:
            self.cfg['keywords'] = {}
        if 'ner' not in self.cfg or self.cfg['ner'] is None:
            self.cfg['ner'] = []
        if 'regex' not in self.cfg or self.cfg['regex'] is None:
            self.cfg['regex'] = {}
        self.replace_blacklist = list(set(list(self.cfg['keywords'].keys()) + self.cfg['ner'] + list(self.cfg['regex'].keys()))) 
        self.build_keywords_regex()


    def __read_keywords(self):
        keywords = {}
        for k in self.cfg['keywords']:
            if not os.path.exists(self.cfg['keywords'][k]):
                continue
            keywords[k] = []
            with open(self.cfg['keywords'][k]) as f:
                for l in f:
                    l = l.strip()
                    if len(l) < 1 or l[0] == '#':
                        continue
                    l_split = [x.strip() for x in re.split(':', l, maxsplit=1)]
                    l_split = [x for x in l_split if len(x) > 0]
                    if len(l_split) < 1:
                        continue
                    entity = l_split[0].lower()
                    keywords[k].append(entity)
                    if len(l_split) == 2 and not entity in self.entity_replace:
                        self.entity_replace[entity] = l_split[1].lower()

            keywords[k] = list(set(keywords[k]))
            keywords[k].sort(key=len, reverse=True)
        return keywords


    def build_keywords_regex(self):
        '''
            build keywords to regex, so the behavior of the keywords recongition will be the same as regex 
        '''
        keywords = self.__read_keywords() 
        for k in keywords:
            keywords_temp = keywords[k]
            for kwd in keywords_temp:
                if kwd in self.entity_replace and not self.entity_replace[kwd] in keywords_temp:
                    keywords_temp.append(self.entity_replace[kwd])
            keywords_temp.sort(key=len, reverse=True)
            keyword = ['('+l+')' for l in keywords_temp]
            self.keywords_regex[k] = '|'.join(keyword)

    
    def build_keywords_index(self, emb_ins):
        '''
            Build the keywords index via annoy search(word vector search). Any available cfg keys please check ailab.text.annoysearch

            Input:
                - emb_ins: instance of ailab.text.embedding
        '''
        from .annoysearch import AnnoySearch
        keywords = self.__read_keywords()
        self.keywords_index = AnnoySearch(self.cfg, emb_ins)
        self.keywords_kw2e = {}
        for k in keywords:
            for kw in keywords[k]:
                self.keywords_kw2e[kw] = k
            
        keywords = list(self.keywords_kw2e.keys())
        keywords.sort(key=len, reverse=True)
        self.keywords_index.load_index(keywords)


    def get_keywords(self, sentence, replace = False, entities=None):
        '''
           get the keyword entities from sentence

           Input:
                - sentence: string or a token list
                - replace: to replace the token to entity name in tokens list, default is False
                - entities: the existed entity dictionary. will append the found entities to this dictionary
            
            Output:
                - if the replace is True, will return (entities, tokens)
                - if the replace is False, will return entities
                - entities: a dictionary of entity with format of {entity name: entitiy value, ...}
                - tokens: the token list
        '''
        if isinstance(sentence, str):
            tokens = self.seg(sentence)['tokens']
        else:
            tokens = sentence
        if entities is None:
            entities = {}
        entity = self.keywords_index.find(tokens, location=True)
        if len(entity) > 0:
            entity, loc = tuple(zip(*entity))
            for i, il in enumerate(loc):
                if tokens[il][0] == '{' and tokens[il][-1] == '}':
                    continue
                k = self.keywords_kw2e[entity[i]]
                if not k in entities:entities[k] = []
                if entity[i] in self.entity_replace:
                    entities[k].append(self.entity_replace[entity[i]])
                else:
                    entities[k].append(entity[i])
                if replace:
                    tokens[il] = '{' + k.upper() + '}'
        if replace:
            return entities, tokens
        else:
            return entities


    #get entities via regex
    def get_regex(self, sentence, replace = False, entities=None):
        '''
            get the entities from regex

            Input:
                - sentence: string
                - replace: to replace the token to entity name in sentence, default is False
                - entities: the existed entity dictionary. will append the found entities to this dictionary
            
            output:
                - if the replace is True, will return (entities, replaced sentence)
                - if the replace is False, will return entities
                - entities: a dictionary of entity with format of {entity name: entitiy value, ...}
                - replaced: replaced sentence (the entity value in sentence will be replaced to entity name)
        '''
        if entities is None:
            entities = {}
        replaced = sentence
        regex = dict(**self.cfg['regex'], **self.custom_regex, **self.keywords_regex)
        for reg in list(self.cfg['regex'].keys()) + list(self.custom_regex.keys()) + list(self.keywords_regex.keys()):
            for entity in re.finditer(regex[reg], replaced):
                if not reg in entities:
                    entities[reg] = []
                e = entity.group(0)
                if e in self.entity_replace:
                    entities[reg].append(self.entity_replace[e])
                else:
                    entities[reg].append(e)
            replaced = re.sub(regex[reg], '{'+reg.upper()+'}', replaced)
        if replace:
            return entities, replaced
        else:
            return entities

    
    def get(self, sentence, entities = None):
        '''
            get the entities from keywords, regex, ner

            Input:
                - sentence: string or token list
                - entities: the existed entity dictionary. will append the found entities to this dictionary
            
            Output: a tuple of (entities, tokens)
                - entities: a dictionary of entity with format of {entity name: entitiy value, ...}
                - tokens: the token list, note the entity values are replaced to entity names
        '''
        if entities is None:
            entities = {}
        entities, replace_regex = self.get_regex(sentence, True, entities)
        tokens = self.seg(replace_regex)
        if self.keywords_index is not None:
            entities, tokens['tokens'] = self.get_keywords(tokens['tokens'], True, entities)
        entities, tokens = self.get_ner(tokens, True, entities)
        return entities, tokens
    
    
    def get_ner(self, sentence, replace = False, entities=None):
        '''
            get the entities viaNER

            Input:
                - sentence: string or token list
                - replace: to replace the token to entity name in tokens list, default is False
                - entities: the existed entity dictionary. will append the found entities to this dictionary
            
            Output:
                - if the replace is True, will return (entities, tokens)
                - if the replace is False, will return entities
                - entities: a dictionary of entity with format of {entity name: entitiy value, ...}
                - tokens: the token list
        '''
        if entities is None:
            entities = {}
        replace_blacklist = list(set(list(entities.keys()) + self.replace_blacklist))
        if isinstance(sentence, str):
            tokens = self.seg(sentence)
        else:
            tokens = sentence
        for i, e in enumerate(tokens['entities']):
            if len(e) > 0 and e in self.cfg['ner'] and  not tokens['tokens'][i] in replace_blacklist:
                if e in entities:
                    entities[e].append(tokens['tokens'][i])
                else:
                    entities[e] = [tokens['tokens'][i]]
        if replace:
            replaced = []
            for i, e in enumerate(tokens['entities']):
                if len(tokens['tokens'][i]) < 1:continue
                elif tokens['tokens'][i] in ['{', '}']:
                    continue
                if i > 0 and i < len(tokens['tokens'])-1 and tokens['tokens'][i-1] == '{' and tokens['tokens'][i+1] == '}':
                    replaced.append('{' + tokens['tokens'][i].upper() + '}')
                elif e not in self.cfg['ner'] or len(e) < 1:
                    replaced.append( tokens['tokens'][i] )
                else:
                    replaced.append( '{' + e.upper() + '}' )
            for i in range(len(replaced)-1, 0, -1):
                if replaced[i] == replaced[i-1]:
                    del replaced[i]
            return entities, replaced

        else:
            return entities


class NER_CoreNLP(NER_Base, Segment_CoreNLP):
    '''
        The NER part uses stanford CoreNLP. The class inherit from NER_Base and ailab.text.Segment_CoreNLP
        
        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: please check the needed keys from NER_Base and ailab.text.Segment_CoreNLP
    '''
    def __init__(self, cfg):
        NER_Base.__init__(self,cfg)
        Segment_CoreNLP.__init__(self, cfg)
    
    def train(self, entities, data, n_iter=50):
        '''
            Train the corenlp user defined model, not implemented
        '''
        raise('The function of CoreNLP ner training is not finished')


class NER_Spacy(NER_Base, Segment_Spacy):
    '''
        The NER part uses Spacy. The class inherit from NER_Base and Spacy
        
        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: please check the needed keys from NER_Base and ailab.text.Segment_Spacy
    '''
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_Spacy.__init__(self, cfg)
        prefix_re = re.compile(r'''^[\‘\[\{\<\(\"\']''')
        suffix_re = re.compile(r'''[\]\}\>\)\"\'\,\.\!\?]$''')
        infix_re = re.compile(r'''[\<\>\{\}\(\)\/\-~\:\,\!\'\’]''')
        from spacy.tokenizer import Tokenizer
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, \
                prefix_search=prefix_re.search, \
                suffix_search=suffix_re.search, \
                infix_finditer=infix_re.finditer)

    def train(self, entities, data, n_iter = 50):
        '''
            Train user defined NER model via spacy api.
            
            Input:
                - entities: entity name list
                - data: format of [(text, annotations), ...]. Please check `spacy document <https://spacy.io/usage/training>`_ for more details

        '''
        import random
        from pathlib import Path
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
        self.nlp.to_disk(self.cfg['cached_ner'])
    

class NER_LTP(NER_Base, Segment_LTP):
    '''
        The NER part uses PyLTP. The class inherit from NER_Base and ailab.text.Segment_LTP
        
        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: please check the needed keys from NER_Base and ailab.text.Segment_LTP
    '''
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_LTP.__init__(self, cfg)

    
    def train_predeal(self, data):
        '''
            Predeal the training data to LTP training data format

            Input:
                - data: format of [(text, annotations), ...], same as spacy. Please check `spacy document <https://spacy.io/usage/training>`_ for more details
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
            if entity in self.cfg['ner_name_replace']:
                entity = self.cfg['ner_name_replace'][entity]
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
                - data: the output from train_predeal
                - maxiter: iteration number
        '''
        nfiles = len(glob.glob(self.cfg['ner_model_path']))
        tmp_file = '/tmp/ner_train_' + uuid.uuid4().hex
        tmp_ner_file = tmp_file + '.ner'
        father_dir = os.path.dirname(self.cfg['ner_model_path'])
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


class NER_Rest(NER_Base, Segment_Rest):
    '''
        The NER part uses restapi. The class inherit from NER_Base and ailab.text.Segment_Rest
        
        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: please check the needed keys from NER_Base and ailab.text.Segment_Rest
    '''
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_Rest.__init__(self, cfg)
    
    def train(self, entities, data, n_iter=50):
        '''
            Training the user defined NER model, not implemented.
        '''
        raise('training via NER Rest api is not supported')


class NER(object):
    '''
        Entity recognition tool, integrate with several tools 

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: 
                    - TOKENIZER: string, choose for NER class:
                        1. *corenlp*: will use NER_CoreNLP
                        2. *spacy*: will use NER_Spacy
                        3. *ltp*: will use NER_LTP
                        4. *http://**: will use NER_Rest
                    - LANGUAGE: if *TOKENIZER* not exists, will check the *LANGUAGE* config:
                        1. cn: will use NER_LTP
                        2. en: will use NER_Spacy

        Example Usage:
            - ner = NER(cfg); ner.get(sentence)
    '''
    def __new__(cls, cfg):
        tokenizers = {'corenlp':NER_CoreNLP, \
                      'spacy':NER_Spacy, \
                      'ltp':NER_LTP}
        languages = {'cn':'ltp', \
                     'en':'spacy'}
        if 'TOKENIZER' in cfg:
            if cfg['TOKENIZER'] in tokenizers:
                return tokenizers[cfg['TOKENIZER']](cfg)
            elif 'http' in cfg['TOKENIZER']:
                return NER_Rest(cfg) 
        if 'LANGUAGE' in cfg and cfg['LANGUAGE'] in languages:
            return tokenizers[languages[cfg['LANGUAGE']]](cfg)
        raise('Error! %s language is not supported'%cfg['LANGUAGE'])



