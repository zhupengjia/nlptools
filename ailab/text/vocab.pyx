#!/usr/bin/env python
import numpy, os
from ..utils import zload, zdump
from sklearn.utils import murmurhash3_32

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
        if 'vocab_hash_size' in self.cfg:
            self.vocab_hash_size = self.cfg['vocab_hash_size']
        else:
            self.vocab_hash_size = 24

    def __get_cached_vocab(self):
        if os.path.exists(self.cfg['cached_vocab']):
            cached_vocab = zload(self.cfg['cached_vocab'])
            self._id2word, self._id2tf, self._id2vec, self._has_vec  = cached_vocab
        else:
            self._id2word = {}
            self._id2tf = {}
            self._id2vec = numpy.zeros((2**self.vocab_hash_size, self.emb_ins.vec_len), 'float32')
            self._has_vec = numpy.zeros(2**self.vocab_hash_size, numpy.bool_)
    
    def save(self):
        zdump((self._id2word, self._id2tf, self._id2vec, self._has_vec), self.cfg['cached_vocab'])

    @staticmethod
    def hashword(word, hashsize=24):
        return murmurhash3_32(word, positive=True) % (2**hashsize)

    @property
    def Nwords(self):
        return len(self._id2word)

    def __len__(self):
        return len(self._id2word)
    
    def add_word(self, word):
        wordid = Vocab.hashword(word, self.vocab_hash_size) 
        if wordid not in self._id2word:
            self._id2word[wordid] = word
            self._id2tf[wordid] = 1
        else:
            self._id2tf[wordid] += 1
        return wordid

    def word2id(self, word, fulfill=True):
        wordid = Vocab.hashword(word, self.vocab_hash_size) 
        if wordid not in self._id2word:
            if fulfill:
                self._id2word[wordid] = word
                self._id2tf[wordid] = 1
            else:
                return None
        return wordid

    def sentence2id(self, sentence):
        ids = []
        if not any([isinstance(sentence, list), isinstance(sentence, tuple)]):
            sentence = self.seg_ins.seg_sentence(sentence)
        hash_sentence = hash(''.join(sentence))
        if hash_sentence in self.sentences_hash:
            func_add_word = self.word2id
        else:
            func_add_word = self.add_word
        self.sentences_hash[hash_sentence] = 0
        for t in sentence['tokens']:
            ids.append(func_add_word(t))
        return ids
    
    def get_id2vec(self):
        len_id2vec = len(self._has_vec[self._has_vec])
        for i in self._id2word:
            if not self._has_vec[i]:
                self._id2vec[i] = self.emb_ins[self._id2word[i]]
        return len(self._id2word) - len_id2vec

    def senid2tf(self, sentence_id):
        return [self._id2tf[x] for x in sentence_id]

    def senid2vec(self, sentence_id):
        vec = numpy.zeros((len(sentence_id), self.emb_ins.vec_len), 'float32')
        for i,sid in enumerate(sentence_id):
            vec[i] = self.emb_ins[self._id2word[sid]]
        return vec
    
    def word2vec(self, word_id):
        return self.emb_ins[self._id2word[word_id]]
    
    def ave_vec(self, sentence_id):
        vec = numpy.zeros(self.emb_ins.vec_len, 'float32')
        tottf = 0
        for i, sid in enumerate(sentence_id):
            w = numpy.log(self.Nwords/(self._id2tf[sid]))
            vec += self.emb_ins[self._id2word[sid]]*w
            tottf += w
        if tottf == 0:
            return numpy.zeros(self.emb_ins.vec_len)
        return vec/tottf


