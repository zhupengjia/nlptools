#!/usr/bin/env python
import numpy, os, random
import unicodedata
from .tokenizer import Segment
from ..utils import zload, zdump, hashword, normalize


# get TF of vocabs and vectors
class Vocab(object):
    def __init__(self, cfg={}, seg_ins=None, emb_ins=None, forceinit=False):
        self.cfg = {'cached_vocab': '', 'vocab_size': 2**15, 'ngrams':1, 'outofvocab':'unk', 'hashgenerate':0}
        for k in cfg: self.cfg[k] = cfg[k]
        self.seg_ins = seg_ins
        self.seg_ins_emb = seg_ins is None # if tokenizer embedded in Vocab or as a parameter input
        self.emb_ins = emb_ins
        self.sentences_hash = {} #check if sentence added
        self.vocab_size = int(self.cfg['vocab_size'])
        if self.vocab_size < 30:
            self.vocab_size = 2**int(self.cfg['vocab_size'])
        self.__get_cached_vocab(forceinit)

    def __del__(self):
        del self._id2word, self._word2id, self._id2tf, self._id_ngrams
        if self.seg_ins_emb: 
            del self.seg_ins

    def addBE(self):
        self._word_spec = ['<pad>', '<eos>', '<bos>','<unk>']
        self._id_spec = [self.word2id(w) for w in self._word_spec]
        self.PAD, self.EOS, self.BOS, self.UNK = tuple(self._word_spec)
        self._id_PAD, self._id_EOS, self._id_BOS, self._id_UNK = tuple(self._id_spec)

    def __get_cached_vocab(self, forceinit):
        ifinit = True
        self._id_UNK = 0
        if os.path.exists(self.cfg['cached_vocab']) and not forceinit:
            try:
                cached_vocab = zload(self.cfg['cached_vocab'])
                self._id2word, self._word2id, self._id2tf, self._id_ngrams  = cached_vocab
                if len(self._id2word) > 0:
                    self._vocab_max = max(self._id2word)
                    ifinit = False
            except Exception as err:
                print('warning!!! cached vocab read failed!!! ' + err)
        if ifinit:
            self._id2word = {}
            self._word2id = {}
            self._id2tf = numpy.zeros(self.vocab_size, 'int')
            self._vocab_max = -1
            self._id_ngrams = {}
        if self.cfg['ngrams']>1:
            self.addBE()

    def save(self):
        if len(self.cfg['cached_vocab']) > 0:
            zdump((self._id2word, self._word2id, self._id2tf, self._id_ngrams), self.cfg['cached_vocab'])


    def accumword(self, word, fulfill=True, tfaccum = True):
        if word in self._word2id:
            if tfaccum: self._id2tf[self._word2id[word]] += 1
            return self._word2id[word]
        elif fulfill:
            if self.cfg['hashgenerate']:
                wordid = hashword(word, self.vocab_size)
                if wordid > self._vocab_max:
                    self._vocab_max = wordid
                self._id2word[wordid] = word
                self._id2tf[wordid] = 1
            else:
                if self._vocab_max < self.vocab_size - 1:
                    self._vocab_max += 1
                    wordid = self._vocab_max
                    self._id2word[wordid] = word
                    self._id2tf[wordid] = 1
                else:
                    wordid = self._id_UNK
                    self._id2tf[wordid] += 1
            self._word2id[word] = wordid
            return wordid
        else:
            return None


    @property
    def Nwords(self):
        return len(self._id2word)


    def __len__(self):
        return len(self._id2word)

    #reduce vocab size by tf
    def reduce_vocab(self, vocab_size=None):
        if vocab_size is None:
            self.vocab_size = self._vocab_max + 1
            self._id2tf = self._id2tf[:self.vocab_size]
            return
        elif vocab_size >= self.vocab_size:
            return
        new_id2word = {}
        new_word2id = {}
        new_id2tf = numpy.zeros(self.vocab_size, 'int')
        new_id_ngrams = {}
        id_mapping = {}
        if self.cfg['hashgenerate']:
            for old_id in self._id2word:
                word = self._id2word[old_id]
                new_id = old_id % self.vocab_size
                new_id2word[new_id] = word
                new_word2id[word] = new_id
                new_id2tf[new_id] = self._id2tf[old_id]
            for old_id in self._id_ngrams:
                new_id = old_id % self.vocab_size
                new_id_ngrams[new_id] = [i % self.vocab_size for i in self._id_ngrams[old_id]]
        else:
            maxtf = self._id2tf.max()
            for i in range(len(self._id_spec)):
                self._id2tf[self._id_spec[i]] = maxtf + len(self._id_spec) - i #avoid remove spec vocab
            sortedid = numpy.argsort(self._id2tf)[::-1]
            new_id_N = 0
            for old_id in sortedid:
                if not old_id in self._id2word: continue
                word = self._id2word[old_id]
                if new_id_N >= vocab_size:
                    if self.cfg['outofvocab']=='random':
                        new_id = random.randint(0, vocab_size-1)
                    else:
                        new_id = self._id_UNK
                else:
                    new_id = new_id_N
                    new_id2tf[new_id] = self._id2tf[old_id]
                    new_id2word[new_id] = word
                    new_id_N += 1
                id_mapping[old_id] = new_id
                new_word2id[word] = new_id
            for old_id in self._id_ngrams:
                new_id_ngrams[id_mapping[old_id]] = [id_mapping[i] for i in self._id_ngrams[old_id]]
        self._id2word = new_id2word
        self._word2id = new_word2id
        self._id2tf = new_id2tf
        self._id_ngrams = new_id_ngrams
        self.vocab_size = vocab_size
        self._vocab_max = max(self._id2word)
    
 
    #call function, convert sentences to id
    def __call__(self, sentences):
        ids = []
        for sentence in sentences:
            ids.append(self.sentence2id(sentence))
        return ids
  
    #get id from word, or word from id
    def __getitem__(self, key):
        if isinstance(key, int):
            if key in self._id2word:
                return self._id2word[key]
            else:
                return None
        return self.word2id(key)

    #set id for word or word for id manually, be careful!
    def __setitem__(self, key, item):
        if isinstance(key, int):
            word, wordid = item, key
        else:
            word, wordid = key, item
        wordid_outofvocab = False
        if self.cfg['hashgenerate']:
            wordid = wordid % self.vocab_size
        else:
            if wordid >= self.vocab_size:
                wordid_outofvocab = True
                if self.cfg['outofvocab']=='random':
                    word2id = random.randint(0, self.vocab_size-1)
                else:
                    word2id = self._id_UNK
                wordid = self._id_UNK
        if wordid > self._vocab_max:
            self._vocab_max = wordid
        self._word2id[word] = wordid
        if not wordid_outofvocab:
            self._id2word[wordid] = word
            self._id2tf[wordid] = 0

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self._id2word
        return key in self._word2id

    def add_word(self, word):
        return self.accumword(word)


    def word2id(self, word, fulfill=True):
        return self.accumword(word, fulfill, False)


    #sentence to vocab id, useBE is the switch for adding BOS and EOS in prefix and suffix
    def sentence2id(self, sentence, ngrams=None, useBE=True, addforce=True):
        if isinstance(sentence, str):
            if self.seg_ins is None:
                self.seg_ins = Segment(self.cfg)
            sentence_seg = self.seg_ins.seg(sentence)['tokens']
        elif numpy.isnan(sentence):
            return []
        else:
            sentence_seg = sentence
            sentence = ''.join(sentence_seg)

        if ngrams is None:
            ngrams = self.cfg['ngrams']
        hash_sentence = hash(sentence)
        if not hash_sentence in self.sentences_hash or addforce:
            func_add_word = self.add_word
        else:
            func_add_word = self.word2id

        self.sentences_hash[hash_sentence] = 0
        ids = [func_add_word(t) for t in sentence_seg]
        #ngrams
        if ngrams > 1:
            if useBE:
                ids_BE = [self._id_BOS] + ids + [self._id_EOS]
                words_BE = [self.BOS] + sentence_seg + [self.EOS]
            else:
                ids_BE = ids
                words_BE = sentence_seg
        for n in range(1, ngrams):
            if len(ids_BE) < n: break
            ids_gram_tuple = [ids_BE[i:i+n+1] for i in range(len(ids_BE)-n)]
            ids_gram_word = [''.join(words_BE[i:i+n+1]) for i in range(len(words_BE)-n)]
            ids_gram = [func_add_word(x) for x in ids_gram_word]
            ids += ids_gram
            for d in zip(ids_gram, ids_gram_tuple):
                if not d[0] in self._id_ngrams:
                    self._id_ngrams[d[0]] = d[1]
        return ids

    #used to cache word2vec
    def get_id2vec(self):
        if self.emb_ins is None:
            return None
        for i in self._id2word:
            self.id2vec(i)
    
    #return dense vector for id2vec
    def dense_vectors(self):
        self.get_id2vec()
        vectors = numpy.zeros((self._vocab_max+1, self.emb_ins.vec_len), 'float')
        for k in self._id2word:
            if k == self._id_PAD:
                vectors[k] = 0
            else:
                vectors[k] = self.id2vec(k)
        return vectors


    def senid2tf(self, sentence_id):
        return [self._id2tf[x] for x in sentence_id]


    def senid2vec(self, sentence_id):
        if self.emb_ins is None:
            return None
        vec = numpy.zeros((len(sentence_id), self.emb_ins.vec_len), 'float')
        for i,sid in enumerate(sentence_id):
            vec[i] = self.id2vec(sid)
        return vec


    def id2vec(self, word_id):
        if word_id in self._id_ngrams:
            return numpy.sum([self.emb_ins[self._id2word[ii]] for ii in self._id_ngrams[word_id]], axis=0)
        else:
            return self.emb_ins[self._id2word[word_id]]


    def ave_vec(self, sentence_id):
        vec = numpy.zeros(self.emb_ins.vec_len, 'float')
        tottf = 0
        for i, sid in enumerate(sentence_id):
            w = numpy.log(self.Nwords/(self._id2tf[sid]))
            vec += self.emb_ins[self._id2word[sid]]*w
            tottf += w
        if tottf == 0:
            return numpy.zeros(self.emb_ins.vec_len)
        return vec/tottf




