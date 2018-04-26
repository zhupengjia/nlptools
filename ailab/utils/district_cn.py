#!/usr/bin/env python
import plistlib, re, os, json, sys, time
import xml.etree.ElementTree as etree
from treelib import Tree, Node
from .utils import zdump, zload, flat_list
from ..text.vectfidf import VecTFIDF
from ..text.tokenizer import Segment_Char
from ..text.vocab import Vocab

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class District_CN:
    '''
        get chinese district 

        input:
            - dbdir: database location
    '''
    def __init__(self, dbdir):
        self.dbdir = dbdir
        self.cfg = {'TOKENIZER': 'char', \
                'ngrams':3,\
                'cached_index': os.path.join(dbdir, 'index.pkl'),\
                'cached_vocab': os.path.join(dbdir, 'vocab.pkl'),\
                'cached_data': os.path.join(dbdir, 'district.pkl'),\
                'plist': os.path.join(dbdir, 'list.plist')}
        self.__build()


    def __build(self):
        '''
            load index
        '''
        self.tokenizer = Segment_Char(self.cfg) 
        self.vocab = Vocab(self.cfg, self.tokenizer)
        self.vocab.addBE()
        self.index = VecTFIDF(self.cfg, self.vocab)
        if os.path.exists(self.cfg['cached_data']):
            data = zload(self.cfg['cached_data'])
            self.tree = data['tree']
            self.tags_id = data['tags_id']
            self.tags_name = data['tags_name']
            self.index.load_index()
            return 
        self.tree = Tree()
        data = plistlib.readPlist(self.cfg['plist'])
        self.tags = {}
        self.tree.create_node("address", 0)
        self.__loopdata(data, 0)
        
        #index
        tags_zip = list(zip(*self.tags.items()))
        self.tags_name = tags_zip[0]
        self.tags_id = tags_zip[1]
        tags_wordids = []
        
        self.index = VecTFIDF(self.cfg, self.vocab)
        tags_wordids = [self.__sentence2id(x) for x in self.tags_name]
        
        self.index.load_index(tags_wordids)
        self.vocab.save()
        zdump({"tree":self.tree, "tags_id":self.tags_id, "tags_name":self.tags_name}, self.cfg['cached_data'])

    
    def __sentence2id(self, sentence):
        ids = self.vocab.sentence2id(sentence, useBE=False, remove_stopwords=False, flatresult = False)
        #only keep 2-gram, 3-gram
        ids = flat_list([ids[i] for i in [2, 3] if i in ids])
        return ids


    def __loopdata(self, nodes, father=None):
        for k in nodes:
            nid = int(re.findall('\d+',k)[0])
            nname = re.findall('\D+', k)[0]
            self.tree.create_node(nname, nid, parent=father)
            if not nname in self.tags:
                self.tags[nname] = []
            self.tags[nname].append(nid)
            if not isinstance(nodes, dict):
                childjson = 'town/{}.json'.format(nid)
                if os.path.exists(childjson):
                    with open(childjson) as f:
                        childdata = json.load(f)
                    childdata = [''.join(x) for x in childdata.items()]
                    self.__loopdata(childdata, nid)
                   
            elif isinstance(nodes[k], (dict, list)):
                self.__loopdata(nodes[k], nid)


    def get_nodes(self, name, topN=1):
        '''
            get node names

            input:
                - name: node name
        '''
        name_id = self.__sentence2id(name)
        if len(name_id) < 1:
            return []
        index_ids = self.index.search_index(name_id, topN=topN)
        ids = [x[0] for x in index_ids]
        names = {}
        for i in ids:
            tagname = self.tags_name[i]
            tagid = self.tags_id[i]
            subtree = self.tree.rsearch(tagid)
            names[subtree.DEPTH] = tagname
        return names   
   






