#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
from .model import DocReader
from ailab.utils import zload, zdump, normalize
from ailab.text import *
import torch, json, os, re

class DrQA(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.emb_ins = Embedding(cfg)
        self.seg_ins = Segment(cfg)
        self.vocab = Vocab(cfg, self.seg_ins, self.emb_ins, 3, hashgenerate=False)
        self.vocab.addBE(forceadd=True)
        self.tfidf = VecTFIDF(cfg, self.vocab)
   
    #load reader model
    def load_reader(self):
        self.reader_params = DocReader.load(self.cfg['drqa_reader_path'])
        #vocab merge
        for w in self.reader_params['word_dict']:
            self.vocab[w] = self.reader_params['word_dict'][w]
        #cfg merge
        for k in self.reader_params['args'].__dict__:
            self.cfg[k] = self.reader_params['args'].__dict__[k]
        self.reader = DocReader(self.cfg, self.vocab, self.reader_params['feature_dict'], self.reader_params['state_dict'])

    def train(self, documents):
        #predeal
        if 'drqa_data_cache' in self.cfg and os.path.exists(self.cfg['drqa_data_cache']):
            self.corpus, self.corpus_segs, corpus_ids = zload(self.cfg['drqa_data_cache'])
        else:
            self.corpus, self.corpus_segs, corpus_ids = [], [], []
            for text in documents:
                text = [x for x in re.split('[\n\.\[\]]', text) if len(x)>2]
                text_seg = [self.seg_ins.seg(x) for x in text]
                text_ids = [self.vocab.sentence2id(x['tokens']) for x in text_seg]
                self.corpus += text
                self.corpus_segs += text_seg
                corpus_ids += text_ids
        #tfidf training
        self.tfidf.load_index(corpus_ids) 
    
    def search(self, query, topN=1):
        query_seg = self.seg_ins.seg(query)
        query_id = self.vocab.sentence2id(query_seg['tokens'])
        query_seg['id'] = torch.LongTensor(self.vocab.sentence2id(query_seg['tokens'], 1))

        ranked_doc = self.tfidf.search_index(query_id, topN*5)
        ranked_doc_indexes = list(zip(*ranked_doc))[0]
        ranked_doc = [self.corpus[i] for i in ranked_doc_indexes]
        ranked_doc_seg = [self.corpus_segs[i] for i in ranked_doc_indexes]
        
        for i in range(len(ranked_doc_seg)):
            ranked_doc_seg[i]['id'] = torch.LongTensor(self.vocab.sentence2id(ranked_doc_seg[i]['tokens'], 1))
        
        examples = []
        for i in range(len(ranked_doc_seg)):
            examples.append(self.vectorize(i, query_seg, ranked_doc_seg[i]))
        #build the batch and run it through the mode
        batch_exs = self.batchify(examples)
        s, e, score = self.reader.predict(batch_exs, None, topN)
        print(s)
        print(e)
        print(score)
        
        #retrieve the predicted spans
        results = []
        #for i in range(len(s)):
        #    predictions = []
        #    for j in range(len(s[i])):
        #        span = d_tokens[i].slice(s[i][j], e[i][j]+1)
    


    def vectorize(self, eid, query, documents):
        # Create extra features vector
        if len(self.reader_params['feature_dict']) > 0:
            features = torch.zeros(len(documents['id']), len(self.reader_params['feature_dict']))
        else:
            features = None
    
        # f_{exact_match}
        if self.cfg['use_in_question']:
            q_words_cased = {w for w in query['texts']}
            q_words_uncased = {w.lower() for w in query['texts']}
            q_lemma = {w for w in query['tokens']} if self.cfg['use_lemma'] else None
            for i in range(len(documents['texts'])):
                if documents['texts'][i] in q_words_cased:
                    features[i][self.reader_params['feature_dict']['in_question']] = 1.0
                if documents['texts'][i].lower() in q_words_uncased:
                    features[i][self.reader_params['feature_dict']['in_question_uncased']] = 1.0
                if q_lemma and documents['tokens'][i] in q_lemma:
                    features[i][self.reader_params['feature_dict']['in_question_lemma']] = 1.0
    
        # f_{token} (POS)
        if self.cfg['use_pos']:
            for i, w in enumerate(documents['pos']):
                f = 'pos=%s' % w
                if f in self.reader_params['feature_dict']:
                    features[i][self.reader_params['feature_dict'][f]] = 1.0
    
        # f_{token} (NER)
        if self.cfg['use_ner']:
            for i, w in enumerate(documents['entities']):
                f = 'ner=%s' % w
                if f in self.reader_params['feature_dict']:
                    features[i][self.reader_params['feature_dict'][f]] = 1.0
    
        # f_{token} (TF)
        if self.cfg['use_tf']:
            counter = Counter([w.lower() for w in documents['texts']])
            l = len(documents['texts'])
            for i, w in enumerate(documents['texts']):
                features[i][self.reader_params['feature_dict']['tf']] = counter[w.lower()] * 1.0 / l
    
        return documents['id'], features, query['id'], eid


    def batchify(self, batch):
        """Gather a batch of individual examples into one batch."""
        NUM_INPUTS = 3
        NUM_TARGETS = 2
        NUM_EXTRA = 1
    
        ids = [ex[-1] for ex in batch]
        docs = [ex[0] for ex in batch]
        features = [ex[1] for ex in batch]
        questions = [ex[2] for ex in batch]
    
        # Batch documents and features
        max_length = max([d.size(0) for d in docs])
        x1 = torch.LongTensor(len(docs), max_length).zero_()
        x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
        if features[0] is None:
            x1_f = None
        else:
            x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
        for i, d in enumerate(docs):
            x1[i, :d.size(0)].copy_(d)
            x1_mask[i, :d.size(0)].fill_(0)
            if x1_f is not None:
                x1_f[i, :d.size(0)].copy_(features[i])
    
        # Batch questions
        max_length = max([q.size(0) for q in questions])
        x2 = torch.LongTensor(len(questions), max_length).zero_()
        x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
        for i, q in enumerate(questions):
            x2[i, :q.size(0)].copy_(q)
            x2_mask[i, :q.size(0)].fill_(0)
    
        # Maybe return without targets
        if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
            return x1, x1_f, x1_mask, x2, x2_mask, ids
    
        elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
            # ...Otherwise add targets
            if torch.is_tensor(batch[0][3]):
                y_s = torch.cat([ex[3] for ex in batch])
                y_e = torch.cat([ex[4] for ex in batch])
            else:
                y_s = [ex[3] for ex in batch]
                y_e = [ex[4] for ex in batch]
        else:
            raise RuntimeError('Incorrect number of inputs per example.')
    
        return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids

