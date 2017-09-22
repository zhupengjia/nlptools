#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os, numpy, re, uuid, shutil, glob
from .tokenizer import Segment_Spacy, Segment_LTP


class NER_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
    

class NER_Spacy(NER_Base, Segment_Spacy):
    def __init__(self, cfg):
        NER_Base.__init__(self, cfg)
        Segment_Spacy.__init__(self, cfg)
   
    def add_entity(self, entities):
        for e in entities:
            self.nlp.entity.add_label(e)

    def train(self, data, iteration = 5):
        from spacy.gold import GoldParse
        idx = numpy.arange(len(data))
        numpy.random.shuffle(idx)
        n = 0

        #add new words to vocab
        for d in data:
            doc = self.nlp.make_doc(d[0])
            for word in doc:
                _ = self.nlp.vocab[word.orth]

        for it in range(iteration):
            for i in idx:
                if n%10000 == 0:
                    print n, len(idx)*iteration
                doc = self.nlp.make_doc(data[i][0])
                gold = GoldParse(doc, entities=data[i][1])
                self.nlp.tagger(doc)
                self.nlp.entity.update(doc, gold)
                n += 1
        self.nlp.end_training()
        self.nlp.save_to_directory(self.cfg['cached_ner'])


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
        
    def train(self, data, maxiter=100):
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


class NER(object):
    def __new__(cls, cfg):
        if cfg['LANGUAGE'] in ['en']:
            return NER_Spacy(cfg)
        elif cfg['LANGUAGE'] == 'cn':
            return NER_LTP(cfg)
        else:
            raise('Error! %s language is not supported'%cfg['LANGUAGE'])

