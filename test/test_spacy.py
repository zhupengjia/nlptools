#!/usr/bin/env python3
from nlptools.text.ner import NER_Spacy
from nlptools.utils import zload
import sys

cfg = {
    'cached_ner':'/Users/caozx/daynote/spacy_train_test/',
#    'pos_model_path':'/home/pzhu/data/ltp_data/pos.model',
#    'ner_model_path': '/home/pzhu/data/ltp_data/ner/*.ner',
    'LANGUAGE':'en',
    'entities':'ANIMAL',
    'data':
       [('Horses are too tall and they pretend to care about your feelings',{'entities': [(0,6,'ANIMAL')]}),
		('horses are too tall and they pretend to care about your feelings',{'entities': [(0,6,'ANIMAL')]}),
		('horses pretend to care about your feelings',{'entities': [(0,6,'ANIMAL')]}),
		('Do they bite?', {'entities':[]}),
		('they pretend to care about your feelings, those horses',{'entities': [(48,54,'ANIMAL')]}),
		('horses?',{'entities': [(0,6,'ANIMAL')]})],
    'ner_name_replace':{
        'Nh': 'PERSON',
        'Ni': 'ORG',
        'Ns': 'LOC',
        'person_name': 'PERSON',
        'company_name': 'ORG',
        'product_name': 'PRODUCT',
        'org_name': 'NORP',
        'time': 'TIME'
    }
}

s = NER_Spacy(cfg)
#s.add_entity(cfg['entities'])
s.train(entities=cfg['entities'], data=cfg['data'])
print(s.seg(sys.argv[1]))

#print(s.seg('浙江在线杭州4月25日讯（记者施宇翔 通讯员 方英）毒贩很“时髦”，用微信交易毒品', entityjoin=True))

#sys.exit()

#data = zload('/home/pzhu/data/ner/boson.pkl')

#print(data['entities'])
#s.train(data['train_data'])

