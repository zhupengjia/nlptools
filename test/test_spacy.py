#!/usr/bin/env python3
from ailab.text.ner import NER_LTP
from ailab.utils import zload
import sys

cfg = {
    'cws_model_path':'/home/pzhu/data/ltp_data/cws.model',
    'pos_model_path':'/home/pzhu/data/ltp_data/pos.model',
    'ner_model_path': '/home/pzhu/data/ltp_data/ner/*.ner',
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

s = NER_LTP(cfg)
print(s.seg(sys.argv[1], entityjoin=True))
#print(s.seg('浙江在线杭州4月25日讯（记者施宇翔 通讯员 方英）毒贩很“时髦”，用微信交易毒品', entityjoin=True))

#sys.exit()

#data = zload('/home/pzhu/data/ner/boson.pkl')

#print(data['entities'])
#s.train(data['train_data'])

