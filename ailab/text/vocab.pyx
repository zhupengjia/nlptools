#!/usr/bin/env python
import numpy, os
from multiprocessing import cpu_count, Process ,Pipe, Pool
from .tokenizer import Segment
from ..utils import zload, zdump, hashword
from functools import partial

def sentence2id(cfg, ngrams, sentences):
    tokenize = Segment(cfg)
    ids = []
    for s in sentences:
        words = tokenize(s)['tokens']
        if len(words) < 1:
            continue
        words_BE = ['BOS'] + words + ['EOS']
        for n in range(1, ngrams):
            words += [''.join(words_BE[i:i+n+1]) for i in range(len(words_BE) - n)]
        ids.append([hashword(w, cfg['vocab_hash_size']) for w in words])
    return ids


# get TF of vocabs and vectors
class Vocab:
    def __init__(self, cfg={}, seg_ins=None, emb_ins=None, ngrams=1, hashgenerate=True):
        self.cfg = cfg
        if not 'cached_vocab' in self.cfg:
            self.cfg['cached_vocab'] = ''
        self.seg_ins = seg_ins
        self.emb_ins = emb_ins
        self.sentences_hash = {} #check if sentence added
        self.ngrams = ngrams
        if 'vocab_hash_size' in self.cfg:
            self.vocab_hash_size = 2**self.cfg['vocab_hash_size']
            self.hashgenerate = True
        else:
            self.vocab_hash_size = 2**15
            self.hashgenerate = hashgenerate
        self.__get_cached_vocab()
        self.hashgenerate = hashgenerate

    def __get_cached_vocab(self):
        ifinit = True
        if os.path.exists(self.cfg['cached_vocab']):
            try:
                cached_vocab = zload(self.cfg['cached_vocab'])
                self._id2word, self._word2id, self._id2tf, self._id2vec, self._id_ngrams, self._has_vec  = cached_vocab
                ifinit = False
            except Exception as err:
                print('warning!!! cached vocab read failed!!! ' + err)
        if ifinit:
            self._id2word = {}
            self._word2id = {}
            self._id2tf = {}
            self._id_ngrams = {}
            self._id2vec, self._has_vec = None, None
            if self.hashgenerate:
                if self.emb_ins is not None: 
                    self._id2vec = numpy.zeros((self.vocab_hash_size, self.emb_ins.vec_len), 'float32')
                    self._has_vec = numpy.zeros(self.vocab_hash_size, numpy.bool_)
            else:
                if self.emb_ins is not None:
                    self.id2vec = numpy.zeros((0, self.emb_ins.vec_len), 'float32')
                else:
                    self.id2vec = []
            if self.ngrams>1:
                self._id_BOS = self.word2id('BOS')
                self._id_EOS = self.word2id('EOS')
    
    def save(self):
        if len(self.cfg['cached_vocab']) > 0:
            zdump((self._id2word, self._word2id, self._id2tf, self._id2vec, self._id_ngrams, self._has_vec), self.cfg['cached_vocab'])


    def accumword(self, word, fulfill=True, tfaccum = True):
        if word in self._word2id:
            if tfaccum: self._id2tf[self._word2id[word]] += 1
            return self._word2id[word]
        elif fulfill:
            if self.hashgenerate:
                wordid = hashword(word, self.vocab_hash_size)
            else:
                wordid = len(self._id2word)
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
    
    def add_word(self, word):
        return self.accumword(word)

    def word2id(self, word, fulfill=True):
        return self.accumword(word, fulfill, False)

    #sentence to vocab id, useBE is the switch for adding BOS and EOS in prefix and suffix
    def sentence2id(self, sentence, useBE=False, addforce=True):
        if not any([isinstance(sentence, list), isinstance(sentence, tuple)]):
            sentence = self.seg_ins.seg_sentence(sentence)['tokens']
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
        len_id2vec = len(self._has_vec[self._has_vec])
        if self.hashgenerate:
            for i in self._id2word:
                if not self._has_vec[i]:
                    self._id2vec[i] = self.word2vec(i)
                    self._has_vec[i] = True
        else:
            self._id2vec = numpy.concatenate((self._id2vec, numpy.zeros((len(self._id2word)-self._id2vec.shape[0], self.emb_ins.vec_len)))) 
            for i in range(len_id2vec, len(self._id2word)):
                self._id2vec[i] = self.emb_ins[self._id2word[i]]
        return len(self._id2word) - len_id2vec

    def senid2tf(self, sentence_id):
        return [self._id2tf[x] for x in sentence_id]

    def senid2vec(self, sentence_id):
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


    #document to vocab id(multiprocess), input is sentences. Only support hash for wordid, tf calculate not supported
    def doc2ids(self, sentences):
        #cpus = min(cpu_count() - 2, 4)
        cpus = 1
        step = max(int(len(sentences) / cpus), 1)
        batches = [sentences[i:i + step] for i in range(0, len(sentences), step)]
        
        _sen2id = partial(sentence2id, self.cfg, self.ngrams)
       
        pool = Pool(cpus)
        doc_ids = list(pool.imap_unordered(_sen2id, batches))
        

        #print(doc_ids)



