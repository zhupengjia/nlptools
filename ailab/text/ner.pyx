#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os, numpy, re, uuid, shutil, glob, sys
from .tokenizer import *


class NER_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.custom_regex = {}
        self.keywords_regex = {}
        if 'keywords' not in self.cfg or self.cfg['keywords'] is None:
            self.cfg['keywords'] = {}
        if 'ner' not in self.cfg or self.cfg['ner'] is None:
            self.cfg['ner'] = []
        if 'regex' not in self.cfg or self.cfg['regex'] is None:
            self.cfg['regex'] = {}
        self.replace_blacklist = list(set(list(self.cfg['keywords'].keys()) + self.cfg['ner'] + list(self.cfg['regex'].keys()))) 
        self.get_keywords()
   
    def get_keywords(self):
        for k in self.cfg['keywords']:
            if not os.path.exists(self.cfg['keywords'][k]):
                continue
            with open(self.cfg['keywords'][k]) as f:
                keywords = [l.strip() for l in f]
                keywords = list(set([l.lower() for l in keywords if len(l) > 0]))
                keywords.sort(key=len, reverse=True)
                keywords = ['('+l+')' for l in keywords]
                self.keywords_regex[k] = '|'.join(keywords)

    def get_regex(self, sentence, replace = False, entities=None):
        if entities is None:
            entities = {}
        replaced = sentence
        regex = dict(**self.keywords_regex, **self.cfg['regex'], **self.custom_regex)
        for reg in list(regex.keys()):
            for entity in re.finditer(regex[reg], replaced):
                if reg in entities:
                    entities[reg].append(entity.group(0))
                else:
                    entities[reg] = [entity.group(0)]
            replaced = re.sub(regex[reg], '{'+reg.upper()+'}', replaced)
        if replace:
            return entities, replaced
        else:
            return entities

    
    def get(self, sentence, entities = None):
        if entities is None:
            entities = {}
        entities, replace_regex = self.get_regex(sentence, True, entities)
        entities, tokens = self.get_ner(replace_regex, True, entities)
        return entities, tokens
    
    
    def get_ner(self, sentence, replace = False, entities=None):
        if entities is None:
            entities = {}
        replace_blacklist = list(set(list(entities.keys()) + self.replace_blacklist))
        tokens = self.seg(sentence)
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
    def __init__(self, cfg):
        NER_Base.__init__(self,cfg)
        Segment_CoreNLP.__init__(self, cfg)
    
    def train(self, entities, data, n_iter=50):
        raise('The function of CoreNLP ner training is not finished')


class NER_Spacy(NER_Base, Segment_Spacy):
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
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_LTP.__init__(self, cfg)

    
    def train_predeal(self, data):
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
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_Rest.__init__(self, cfg)
    
    def train(self, entities, data, n_iter=50):
        raise('training via NER Rest api is not supported')


class NER(object):
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



