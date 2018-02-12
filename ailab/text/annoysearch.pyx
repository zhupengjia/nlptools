#!/usr/bin/env python
#search keyword using annoy, via word vectors
import os
#from ..utils import setLogger

class AnnoySearch:
    def __init__(self, cfg, emb_ins, seg_ins = None):
        self.cfg = {'annoy_ntree':10, 'annoy_cache':'', 'annoy_filter':0.5}
        for k in cfg: self.cfg[k] = cfg[k]
        self.emb_ins = emb_ins
        self.seg_ins = seg_ins
        #self.logger = setLogger()

    def load_index(self, keywords):
        from annoy import AnnoyIndex
        self.search = AnnoyIndex(self.emb_ins.vec_len)
        if os.path.exists(self.cfg['annoy_cache']):
            self.search.load(self.cfg['annoy_cache'])
        else:
            self.keywords = keywords
            for i,k in enumerate(keywords):
                self.search.add_item(i,self.emb_ins[k])
            self.search.build(self.cfg['annoy_ntree'])
            if len(self.cfg['annoy_cache']) > 0:
                try:
                    self.search.save(self.cfg['annoy_cache'])
                except Exception as err: 
                    self.logger.warning('Annoy cache failed! ' + err)
    
    def find(self, sentence, remove_stopwords=False, location=False):
        if isinstance(sentence, str):
            if self.seg_ins is None:
                from .tokenizer import Segment
                self.seg_ins = Segment(self.cfg)
            sentence_seg = self.seg_ins.seg(sentence, remove_stopwords=remove_stopwords)['tokens']
        else:
            sentence_seg = sentence
        match = []
        for i, s in enumerate(sentence_seg):
            result = self.search.get_nns_by_vector(self.emb_ins[s], 1, include_distances=True)
            keyword, similarity = result[0][0], 1/(1+result[1][0])
            if similarity < self.cfg['annoy_filter']:
                continue
            if location:
                if isinstance(sentence, str):
                    loc = sentence.index(s)
                else:
                    loc = i
                match.append((self.keywords[keyword], loc))
            else:
                match.append(self.keywords[keyword])
        return match
            

     
        

