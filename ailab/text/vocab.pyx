#!/usr/bin/env python
import numpy, os
import unicodedata
from .tokenizer import Segment
from ..utils import zload, zdump, hashword, normalize


# get TF of vocabs and vectors
class Vocab(object):
    def __init__(self, cfg={}, seg_ins=None, emb_ins=None, ngrams=1, hashgenerate=False, forceinit=False):
        self.cfg = cfg
        if not 'cached_vocab' in self.cfg:
            self.cfg['cached_vocab'] = ''
        self.seg_ins = seg_ins
        self.seg_ins_emb = seg_ins is None # if tokenizer embedded in Vocab or as a parameter input
        self.emb_ins = emb_ins
        self.sentences_hash = {} #check if sentence added
        self.ngrams = ngrams
        if 'vocab_size' in self.cfg:
            self.vocab_size = 2**self.cfg['vocab_size']
        else:
            self.vocab_size = 2**15
        self.hashgenerate = hashgenerate
        self.__get_cached_vocab(forceinit)

    def __del__(self):
        del self._id2word, self._word2id, self._id2tf, self._id2vec, self._id_ngrams, self._has_vec
        if self.seg_ins_emb: 
            del self.seg_ins

    def __addBE(self):
        self._id_BOS = self.word2id('BOS')
        self._id_EOS = self.word2id('EOS')

    def __get_cached_vocab(self, forceinit):
        ifinit = True
        if os.path.exists(self.cfg['cached_vocab']) and not forceinit:
            try:
                cached_vocab = zload(self.cfg['cached_vocab'])
                self._id2word, self._word2id, self._id2tf, self._id2vec, self._id_ngrams, self._has_vec  = cached_vocab
                self._vocab_max = max(self._id2word)
                ifinit = False
            except Exception as err:
                print('warning!!! cached vocab read failed!!! ' + err)
        if ifinit:
            self._id2word = {}
            self._word2id = {}
            self._id2tf = {}
            self._vocab_max = -1
            self._id_ngrams = {}
            self._id2vec, self._has_vec = {}, None
            if self.emb_ins is not None:
                self._has_vec = numpy.zeros(self.vocab_size, numpy.bool_)
        if self.ngrams>1:
            self.__addBE()

    def save(self):
        if len(self.cfg['cached_vocab']) > 0:
            zdump((self._id2word, self._word2id, self._id2tf, self._id2vec, self._id_ngrams, self._has_vec), self.cfg['cached_vocab'])


    def accumword(self, word, fulfill=True, tfaccum = True):
        if word in self._word2id:
            if tfaccum: self._id2tf[self._word2id[word]] += 1
            return self._word2id[word]
        elif fulfill:
            if self.hashgenerate:
                wordid = hashword(word, self.vocab_size)
                if wordid > self._vocab_max:
                    self._vocab_max = wordid
            else:
                if self._vocab_max < self.vocab_size - 1:
                    self._vocab_max += 1
                wordid = self._vocab_max
            self._id2word[wordid] = word
            self._word2id[word] = wordid
            self._id2tf[self._word2id[word]] = 1
            return wordid
        else:
            return None


    @property
    def Nwords(self):
        return len(self._id2word)


    def __len__(self):
        return len(self._id2word)

 
    #call function, convert sentences to id
    def __call__(self, sentences):
        ids = []
        for sentence in sentences:
            ids.append(self.sentence2id(sentence, True))
        return ids
  
    #get id from word, or word from id
    def __getitem__(self, key):
        if isinstance(key, int):
            if key in self._id2word:
                return self._id2word[key]
            else:
                return None
        return self.word2id(key)

    #set id for word or word for id manually
    def __setitem__(self, key, item):
        if isinstance(key, int):
            word, wordid = item, key
        else:
            word, wordid = key, item
        if self.hashgenerate:
            wordid = wordid % self.vocab_size
        else:
            wordid = wordid if wordid < self.vocab_size else self.vocab_size - 1
        if wordid > self._vocab_max:
            self._vocab_max = wordid
        self._id2word[wordid] = word
        self._word2id[word] = wordid
        self._id2tf[wordid] = 0
        if self.emb_ins is not None:
            self._has_vec[wordid] = False

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self._id2word
        return key in self._word2id

    def add_word(self, word):
        return self.accumword(word)


    def word2id(self, word, fulfill=True):
        return self.accumword(word, fulfill, False)


    #sentence to vocab id, useBE is the switch for adding BOS and EOS in prefix and suffix
    def sentence2id(self, sentence, useBE=True, addforce=True):
        if not any([isinstance(sentence, list), isinstance(sentence, tuple)]):
            if self.seg_ins is None:
                self.seg_ins = Segment(self.cfg)
            sentence = self.seg_ins.seg(sentence)['tokens']
        hash_sentence = hash(''.join(sentence))
        if hash_sentence in self.sentences_hash or addforce:
            func_add_word = self.word2id
        else:
            func_add_word = self.add_word

        self.sentences_hash[hash_sentence] = 0
        ids = [func_add_word(t) for t in sentence]
        #ngrams
        if self.ngrams > 1:
            if useBE:
                ids_BE = [self._id_BOS] + ids + [self._id_EOS]
                words_BE = ['BOS'] + sentence + ['EOS']
            else:
                ids_BE = ids
                words_BE = sentence
        for n in range(1, self.ngrams):
            if len(ids_BE) < n: break
            ids_gram_tuple = [ids_BE[i:i+n+1] for i in range(len(ids_BE)-n)]
            ids_gram_word = [''.join(words_BE[i:i+n+1]) for i in range(len(words_BE)-n)]
            ids_gram = [func_add_word(x) for x in ids_gram_word]
            ids += ids_gram
            for d in zip(ids_gram, ids_gram_tuple):
                if not d[0] in self._id_ngrams:
                    self._id_ngrams[d[0]] = d[1]
        return ids


    def get_id2vec(self):
        if self.emb_ins is None:
            return None
        len_id2vec = len(self._has_vec[self._has_vec])
        for i in self._id2word:
            if not self._has_vec[i]:
                self._id2vec[i] = self.word2vec(i)
                self._has_vec[i] = True
        return len(self._id2word) - len_id2vec

    
    #return dense vector for id2vec
    def dense_vectors(self):
        self.get_id2vec()
        vectors = numpy.zeros((self._vocab_max+1, self.emb_ins.__vec_len), 'float32')
        for k in self._id2vec:
            vectors[k] = self._id2vec[k]
        return vectors


    def senid2tf(self, sentence_id):
        return [self._id2tf[x] for x in sentence_id]


    def senid2vec(self, sentence_id):
        if self.emb_ins is None:
            return None
        vec = numpy.zeros((len(sentence_id), self.emb_ins.vec_len), 'float32')
        for i,sid in enumerate(sentence_id):
            vec[i] = self.word2vec(sid)
        return vec


    def word2vec(self, word_id):
        if word_id in self._id_ngrams:
            return numpy.sum([self.emb_ins[self._id2word[ii]] for ii in self._id_ngrams[word_id]], axis=0)
        else:
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




