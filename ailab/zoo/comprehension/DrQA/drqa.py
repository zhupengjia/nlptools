#!/usr/bin/env python3
"""Functions for putting examples into torch format."""

from collections import Counter
from .model import DocReader
from ailab.utils import zload, zdump, normalize
from ailab.text import *
import torch, json, os, re, sys

class DrQA(object):
    def __init__(self, cfg):
        self.cfg_tfidf = cfg['tfidf']
        self.seg_ins = Segment(self.cfg_tfidf)
        self.vocab_tfidf = Vocab(self.cfg_tfidf, self.seg_ins, None, 3, hashgenerate=True)
        self.vocab_tfidf.addBE(forceadd=True)
        self.tfidf = VecTFIDF(cfg, self.vocab_tfidf)
         
        self.cfg_reader = cfg['reader']
        self.emb_ins = Embedding(self.cfg_reader)
        self.vocab_reader = Vocab(self.cfg_reader, self.seg_ins, self.emb_ins, 1, hashgenerate=False)
        self.vocab_reader.addBE(forceadd=True)
   
    #load reader model
    def load_reader(self):
        self.reader_params = DocReader.load(self.cfg_reader['drqa_reader_path'])
        #cfg merge
        for k in self.reader_params['args'].__dict__:
            self.cfg_reader[k] = self.reader_params['args'].__dict__[k]
       
        self.reader = DocReader(self.cfg_reader, self.vocab_reader, self.reader_params['feature_dict'], self.reader_params['state_dict'])

    def build_index(self, documents):
        #predeal
        if 'cached_corpus' in self.cfg_tfidf and os.path.exists(self.cfg_tfidf['cached_corpus']):
            self.corpus, self.corpus_segs, corpus_ids = zload(self.cfg_tfidf['cached_corpus'])
        else:
            self.corpus, self.corpus_segs, corpus_ids = [], [], []
            for text in documents:
                text = [x for x in re.split('\n', text) if len(x)>2]
                text_seg = [self.seg_ins.seg(x) for x in text]
                text_ids = [self.vocab_tfidf.sentence2id(x['tokens']) for x in text_seg]
                self.corpus += text
                self.corpus_segs += text_seg
                corpus_ids += text_ids
            zdump((self.corpus, self.corpus_segs, corpus_ids), self.cfg_tfidf['cached_corpus'])
        #tfidf training
        self.tfidf.load_index(corpus_ids) 
        self.vocab_tfidf.save()

    def tokenize(self, doc):
        doc_seg = self.seg_ins.seg(doc)
        doc_seg['id'] = torch.LongTensor(self.vocab_reader.sentence2id(doc_seg['texts'], addforce=False))
        return doc_seg

    def search(self, query, prefilter = 5, topN=1):
        query_seg = self.tokenize(query)
        query_id = self.vocab_tfidf.sentence2id(query_seg['tokens'])

        ranked_doc = self.tfidf.search_index(query_id, prefilter)

        ranked_doc_indexes = list(zip(*ranked_doc))[0]
        ranked_doc = [self.corpus[i] for i in ranked_doc_indexes]
        ranked_doc_seg = [self.corpus_segs[i] for i in ranked_doc_indexes]
        
        for i in range(len(ranked_doc_seg)):
            ranked_doc_seg[i]['id'] = torch.LongTensor(self.vocab_reader.sentence2id(ranked_doc_seg[i]['texts'], 1, addforce=False))
      
        return self.search_reader(query_seg, ranked_doc_seg)
        
    
    def search_reader(self, query, documents, topN = 1):
        if isinstance(query, str):
            query = self.tokenize(query)
        if isinstance(documents, str):
            documents = [self.tokenize(documents)]
        examples = []

        for i in range(len(documents)):
            examples.append(self.vectorize(i, query, documents[i]))

        #build the batch and run it through the mode
        batch_exs = self.batchify(examples)
        s, e, score = self.reader.predict(batch_exs, None, topN)
       
        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                predictions.append((i, ' '.join(documents[i]['texts'][s[i][j]: e[i][j]+1]), score[i][j]))
            results += predictions
        results.sort(key=lambda r:r[2], reverse=True)
        return results[:topN]


    def vectorize(self, eid, query, documents):
        # Create extra features vector
        if len(self.reader_params['feature_dict']) > 0:
            features = torch.zeros(len(documents['id']), len(self.reader_params['feature_dict']))
        else:
            features = None
    
        # f_{exact_match}
        if self.cfg_reader['use_in_question']:
            q_words_cased = {w for w in query['texts']}
            q_words_uncased = {w.lower() for w in query['texts']}
            q_lemma = {w for w in query['tokens']} if self.cfg_reader['use_lemma'] else None
            for i in range(len(documents['texts'])):
                if documents['texts'][i] in q_words_cased:
                    features[i][self.reader_params['feature_dict']['in_question']] = 1.0
                if documents['texts'][i].lower() in q_words_uncased:
                    features[i][self.reader_params['feature_dict']['in_question_uncased']] = 1.0
                if q_lemma and documents['tokens'][i] in q_lemma:
                    features[i][self.reader_params['feature_dict']['in_question_lemma']] = 1.0
    
        # f_{token} (POS)
        if self.cfg_reader['use_pos']:
            for i, w in enumerate(documents['pos']):
                f = 'pos=%s' % w
                if f in self.reader_params['feature_dict']:
                    features[i][self.reader_params['feature_dict'][f]] = 1.0
    
        # f_{token} (NER)
        if self.cfg_reader['use_ner']:
            for i, w in enumerate(documents['entities']):
                f = 'ner=%s' % w
                if f in self.reader_params['feature_dict']:
                    features[i][self.reader_params['feature_dict'][f]] = 1.0
    
        # f_{token} (TF)
        if self.cfg_reader['use_tf']:
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

