#!/usr/bin/env python
import os, operator, sys, numpy
from .embedding import Embedding_File
from ..utils import zload, distance2similarity

class Synonyms:
    '''
        Get synonyms

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - synonyms_path: path for synonyms index. If not exists, will generate from emb_ins.
                    - w2v_word2idx: path for word-index bidict mapping
                    - synonyms_filter: similarity score filter for finding synonyms, default is 0.5
                    - synonyms_max: max number of synonyms, default is 1000
            - emb_ins: ailab.utils.embedding object, for word2vec source
            - seg_ins: ailab.utils.segment object, for tokenizer

        Special usage:
            - __call__: get synonyms for word
                - input: 
                    - word: string
                    - Nlimit: number limit of returned synonyms, default is 'synonyms_max' in cfg
                    - scorelimit: similarity limit of returned synonyms, default is 'synonyms_filter' in cfg

    '''

    def __init__(self, cfg, emb_ins, seg_ins = None):
        self.cfg = {'synonyms_path':'', 'w2v_word2idx': '', 'synonyms_filter': 0.5, 'synonyms_max': 1000}
        for k in self.cfg: 
            if k in cfg:
                self.cfg[k] = cfg[k]
        self.emb_ins = emb_ins
        self.seg_ins = seg_ins
        self._load_index()


    def _load_index(self):
        '''
            Build or load synonyms index
        '''
        from annoy import AnnoyIndex
        self._search = AnnoyIndex(self.emb_ins.vec_len)
        self._word2idx = zload(self.cfg['w2v_word2idx'])
        if os.path.exists(self.cfg['synonyms_path']):
            self._search.load(self.cfg['synonyms_path'])
        else:
            assert isinstance(self.emb_ins, Embedding_File), 'Word embedding must from file source'
            for word, wordid in sorted(self._word2idx.items(), key=operator.itemgetter(1)):
                if wordid % 10000 == 0 :
                    print('building synonyms index, {}'.format(wordid))
                self._search.add_item(wordid, self.emb_ins[word])
            self._search.build(10)
            if len(self.cfg['synonyms_path']) > 0:
                self._search.save(self.cfg['synonyms_path'])
            
            
    def __call__(self, word, Nlimit = None, scorelimit = None):
        '''
            Looking for synonyms
            
            Input:
                word: string
                Nlimit: number limit of returned synonyms, default is self.cfg['synonyms_max']
                scorelimit: similarity limit of returned synonyms, default is self.cfg['synonyms_filter']
        '''
        if Nlimit is None:
            Nlimit = self.cfg['synonyms_max']
        if scorelimit is None:
            scorelimit = self.cfg['synonyms_filter']
        if word in self._word2idx:
            result, score = self._search.get_nns_by_item(self._word2idx[word], Nlimit, include_distances=True)
        else:
            result, score = self._search.get_nns_by_vector(self.emb_ins[word], Nlimit, include_distances=True)
        result = [self._word2idx.inv[r] for r in result]
        score = distance2similarity(numpy.array(score))
        N_keep = len(score[score > scorelimit])
        result = result[:N_keep]
        score = score[:N_keep]
        return result, score



