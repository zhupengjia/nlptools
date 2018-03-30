#!/usr/bin/env python
import os

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class AnnoySearch:
    '''
        Search keyword using annoy, via word vectors, please check `spotify/annoy <https://github.com/spotify/annoy>`_ for more details

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - annoy_ntree: int, builds a forest of annoy_ntree trees, default is 10
                    - annoy_cache: string, cached index file path, default is ''
                    - annoy_filter: float, similarity threshold, default is 0.5
            - emb_ins: ailab.utils.embedding object, for word2vec source
            - seg_ins: ailab.utils.segment object, for tokenizer
    '''
    def __init__(self, cfg, emb_ins, seg_ins = None):
        self.cfg = {'annoy_ntree':10, 'annoy_cache':'', 'annoy_filter':0.5}
        for k in self.cfg: 
            if k in cfg:
                self.cfg[k] = cfg[k]
        self.emb_ins = emb_ins
        self.seg_ins = seg_ins


    def load_index(self, keywords):
        '''
            Build or load index(if annoy_cache existed)

            Input:
                - keywords: list of string
        '''
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
                    print('Annoy cache failed! ' + err)


    def find(self, sentence, remove_stopwords=False, location=False):
        '''
            Search keywords from sentence

            Input:
                - sentence: string
                - remove_stopwords: bool, if remove stopwords or not, default is False
                - location: bool, if return keyword location or not, default is False

            Output:
                - list, format like [(keyword, location), ...] 

        '''
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
            

     
        

