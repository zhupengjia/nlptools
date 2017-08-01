#!/usr/bin/env python
import numpy, os
from ..utils import zload, zdump

# get TF of vocabs and vectors
class Vocab:
    def __init__(self, cfg={}, seg_ins=None, emb_ins=None):
        self.cfg = cfg
        if not 'cached_vocab' in self.cfg:
            self.cfg['cached_vocab'] = ''
        self.seg_ins = seg_ins
        self.emb_ins = emb_ins
        self.__get_cached_vocab()
        self.sentences_hash = {} #check if sentence added

    def __get_cached_vocab(self):
        if os.path.exists(self.cfg['cached_vocab']):
            cached_vocab = zload(self.cfg['cached_vocab'])
            self._word2id, self.id2word, self.id2tf, self.id2vec, self.Nwords  = cached_vocab
        else:
            self._word2id = {}
            self.id2word = []
            self.id2tf = []
            if self.emb_ins is not None:
                self.id2vec = numpy.zeros((0, self.emb_ins.vec_len), 'float32')
            else:
                self.id2vec = []
            self.Nwords = 0
    
    def save(self):
        zdump((self._word2id, self.id2word, self.id2tf, self.id2vec, self.Nwords), self.cfg['cached_vocab'])

    def add_word(self, word):
        if word not in self._word2id:
            self._word2id[word] = self.Nwords
            self.id2word.append(word)
            self.Nwords += 1
            self.id2tf.append(1)
        else:
            self.id2tf[self._word2id[word]] += 1
        return self._word2id[word]

    def word2id(self, word, fulfill=True):
        if word not in self._word2id:
            if fulfill:
                self._word2id[word] = self.Nwords
                self.id2word.append(word)
                self.Nwords += 1
                self.id2tf.append(1)
            else:
                return None
        return self._word2id[word]
    
    def sentence2id(self, sentence):
        ids = []
        if not any([isinstance(sentence, list), isinstance(sentence, tuple)]):
            sentence = self.seg_ins.seg_sentence(sentence)
        hash_sentence = hash(''.join(sentence))
        if hash_sentence in self.sentences_hash:
            func_add_word = self.word2id
        else:
            func_add_word = self.add_word
        self.sentences_hash[func_add_word] = 0
        for t in sentence['tokens']:
            ids.append(func_add_word(t))
        return ids
    
    def get_id2vec(self):
        len_id2vec = self.id2vec.shape[0]
        self.id2vec = numpy.concatenate((self.id2vec, numpy.zeros((len(self.id2word)-self.id2vec.shape[0], self.emb_ins.vec_len))))
        for i in range(len_id2vec, len(self.id2word)):
            self.id2vec[i] = self.emb_ins[self.id2word[i]]
        return len(self.id2word) - len_id2vec

    def senid2tf(self, sentence_id):
        return [self.id2tf[x] for x in sentence_id]

    def senid2vec(self, sentence_id):
        vec = numpy.zeros((len(sentence_id), self.emb_ins.vec_len), 'float32')
        for i,sid in enumerate(sentence_id):
            vec[i] = self.emb_ins[self.id2word[sid]]
        return vec
    
    def word2vec(self, word_id):
        return self.emb_ins[word_id]
    
    def ave_vec(self, sentence_id):
        vec = numpy.zeros(self.emb_ins.vec_len, 'float32')
        tottf = 0
        for i, sid in enumerate(sentence_id):
            w = numpy.log(self.Nwords/(self.id2tf[sid]))
            vec += self.emb_ins[self.id2word[sid]]*w
            tottf += w
        if tottf == 0:
            return numpy.zeros(self.emb_ins.vec_len)
        return vec/tottf


