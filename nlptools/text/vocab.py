#!/usr/bin/env python
import numpy, os, random, copy
from bidict import bidict
import unicodedata
from .embedding import Embedding_Random
from ..utils import zload, zdump, normalize, flat_list

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

# get TF of vocabs and vectors
class Vocab(object):
    '''
        Vocab dictionary class, also support to accumulate term frequency, return embedding matrix, sentence to id

        Input:
            - cached_vocab: string, cached vocab file path, default is ''
            - vocab_size: int, the dictionary size, default is 1M. If the number<30, then the size is 2**vocab_size, else the size is *vocab_size*.
            - outofvocab: string, the behavior of outofvocab token, default is 'unk'. Two options: 'unk' and 'random'. 'unk' will fill with 'unk', 'random' will fill with a random token
            - embedding: instance of text.embedding, default is None(use random vector)

        Some special operation:
            - __add__: join several vocab together
            - __call__: get ids for sentence list. Input is [sentence, ...], or [token_list, ...]
            - __getitem__: get id for word
            - __len__: get vocab size
            - __contains__: checkout if word or id in vocab
    '''
    def __init__(self, cached_vocab='', vocab_size=1000000, outofvocab='unk', embedding=None):
        self.cached_vocab = cached_vocab
        self.outofvocab = outofvocab
        self.embedding = embedding if embedding is not None else Embedding_Random()
        self.vocab_size = int(self.vocab_size)
        if self.vocab_size < 30:
            self.vocab_size = 2**int(self.vocab_size)
        self.__get_cached_vocab()
        self.update=True


    def addBE(self):
        '''
            Add 4 special characters to dictionary, '<pad>', '<eos>', '<bos>','<unk>'
        '''
        self._word_spec = ['<pad>', '<eos>', '<bos>','<unk>']
        self._id_spec = [self.word2id(w) for w in self._word_spec]
        self.PAD, self.EOS, self.BOS, self.UNK = tuple(self._word_spec)
        self._id_PAD, self._id_EOS, self._id_BOS, self._id_UNK = tuple(self._id_spec)


    def setupdate(self, v=True):
        '''
            Set if vocab need update

            Input:
                - v: bool, True for update, False for freeze
        '''
        self.update = v


    def doc2bow(self, wordlist = None, idlist=None):
        '''
            convert tokens to bow

            Input should be one of them: 
                - wordlist: the token list
                - idlist: the token id list
            
            Output:
                - bow: [(id, tf), ...]
        '''
        if wordlist is None and idlist is None: 
            return []
        if idlist is not None:
            if isinstance(idlist[0], list):
                idlist = flat_list(idlist)
            if isinstance(idlist[0], int):
                ids = idlist
            else:
                raise('idlist has wrong data format!  your input is:' + str(idlist))
        
        else:
            if isinstance(wordlist[0], list):
                wordlist = flat_list(wordlist)
            if isinstance(wordlist[0], str):
                ids = self.words2id(wordlist)
            else:
                raise('docbow input is not supported!  your input is:' + str(wordlist))
        tfs = [self._id2tf[i] for i in ids]
        return list(zip(ids, tfs))


    def __get_cached_vocab(self, forceinit):
        ifinit = True
        self._id_UNK = 0
        if os.path.exists(self.cached_vocab):
            try:
                self._word2id, self._id2tf = zload(self.cached_vocab)
                if len(self._word2id) > 0:
                    ifinit = False
                self.vocab_size = len(self._id2tf)
            except Exception as err:
                print('warning!!! cached vocab read failed!!! ' + err)
        if ifinit:
            self._word2id = bidict()
            self._id2tf = numpy.zeros(self.vocab_size, 'int')


    def save(self):
        '''
            Save the vocab dictionary to *cached_vocab*
        '''
        if len(self.cached_vocab) > 0:
            zdump((self._word2id, self._id2tf), self.cached_vocab)


    def word2id(self, word):
        '''
            Convert word to id

            Input: 
                - word: string
            
            Output:
                - wordid: int    
        '''
        if word is None: return None
        if word in self._word2id:
            if self.update: self._id2tf[self._word2id[word]] += 1
            return self._word2id[word]
        else:
            wordid = len(self._word2id)
            if wordid < self.vocab.size - 1 and self.update:
                self._word2id[word] = wordid
                self._id2tf[wordid] = 1
            else:
               if self.outofvocab=='random':
                   wordid = random.randint(0, self.vocab_size-1)
               else:
                   wordid = self._id_UNK
            return wordid
    
    
    def id2word(self, i):
        '''
            convert id to word.

            Input:
                - i: wordid

            Output:
                - string. If id not in vocab, will return None
        '''
        if i in self._word2id.inv:
            return self._word2id.inv[i]
        return None


    def __len__(self):
        return len(self._word2id)


    def reduce(self, vocab_size=None, reorder=False):
        '''
            reduce vocab size by tf.
            
            Input: 
                - vocab_size: int, the target vocab_size, if None will reduce to the current number of words. Default is None
                - reorder: bool, check if the dictionary reorder by tf. Default is False. Parameter only usable when vocab_size is None. If the vocab_size is not None, the dictionary will always be reordered
        '''
        if vocab_size is None:
            self.vocab_size = len(self._word2id)
            self._id2tf = self._id2tf[:self.vocab_size]
            vocab_size = self.vocab_size
            if not reorder:
                return
        elif vocab_size >= self.vocab_size:
            return
        new_word2id = bidict()
        new_id2tf = numpy.zeros(vocab_size, 'int')
        maxtf = self._id2tf.max()
        for i in range(len(self._id_spec)):
            self._id2tf[self._id_spec[i]] = maxtf + len(self._id_spec) - i #avoid remove spec vocab
        sortedid = numpy.argsort(self._id2tf)[::-1]
        for old_id in sortedid:
            new_id = len(new_word2id)
            if old_id >= len(self._word2id) or new_id >= vocab_size: continue
            word = self._word2id.inv[old_id]
            new_word2id[word] = new_id
            new_id2tf[new_id] = self._id2tf[old_id]

        self._word2id = new_word2id
        self._id2tf = new_id2tf
        self.vocab_size = vocab_size
    

    #merge another vocab
    def __add__(self, other):
        for w in other._word2id:
            _id_other = other._word2id[w]
            if w in self._word2id:
                _id = self._word2id[w]
                self._id2tf[_id] += other._id2tf[_id_other]
            else:
                _id = self.word2id(w)
                self._id2tf[_id] = other._id2tf[_id_other]
        return self


    def __getitem__(self, key):
        '''
            get word from id
        '''
        return self.word2id(key)


    def __contains__(self, word):
        '''
            check if word in vocab

            Input:
                - word: string
        '''
        return key in self._word2id


    def words2id(self, tokens, useBE=True):
        '''
            tokens to token ids
            
            Input:
                - sentence: string or token list
                - useBE: bool, True will add <BOS> and <EOS> for multi-grams ids. False will not. Default is True
                - update: bool, check if use tf accumulate, default is True
                - remove_stopwords: bool, check if remove stopwords, default is True
                - flatresult: bool, check if flat the final result. Please check the output. Default is True

            Output:
                - if flatresult is True, then return a list of ids
                - if flatresult is False, then return a dictionary, key is the n of n-grams and 'char', like 1,2,3,..,,char , value is the word id list.

        '''
        ids = [self.word2id(t) for t in tokens]
        ids = [i for i in ids if i is not None]
        
        return ids


    def id2sentence(self, ids):
        '''
            convert id back to sentence

            Input: 
                - ids: list of ids
        '''
        return ' '.join([self._word2id.inv[i] for i in ids])

    
    def dense_vectors(self):
        '''
            return a numpy array of word vectors. The index of array is the word_ids
        '''
        vectors = numpy.zeros((self.vocab_size, self.embedding.vec_len), 'float')
        for w in self._word2id:
            wordid = self._word2id[w]
            if wordid == 0:
                vectors[wordid] = 0
            else:
                vectors[wordid] = self.embedding[w]

        return vectors


    def senid2tf(self, sentence_id):
        '''
            get counts for each word in token list
            
            Input:
                - sentence_id: token list

            Output:
                - count list
        '''
        return [self._id2tf[x] for x in sentence_id]


    def senid2vec(self, sentence_id):
        '''
            get vord vectors for each word in token list
            
            Input:
                - sentence_id: token list

            Output:
                - numpy array. The array index is the id index in input
        '''
        if self.embedding is None:
            return None
        vec = numpy.zeros((len(sentence_id), self.embedding.vec_len), 'float')
        for i, sid in enumerate(sentence_id):
            vec[i] = self.id2vec(sid)
        return vec


    def id2vec(self, word_id):
        '''
            get the word vector via word_id

            Input:
                - word_id: int

            Output:
                - 1d numpy array. If word_id is multigram's id, will return the sum of the vectors in this gram 
        '''
        return self.embedding[self._word2id.inv[word_id]]


    def ave_vec(self, sentence_id):
        '''
            get tf weighted average word vector of token list

            Input:
                - sentence_id: token_id list
            
            Output:
                - 1d numpy array
        '''
        vec = numpy.zeros(self.embedding.vec_len, 'float')
        tottf = 0
        for i, sid in enumerate(sentence_id):
            w = numpy.log(self.Nwords/(self._id2tf[sid]))
            vec += self.embedding[self._word2id.inv[sid]]*w
            tottf += w
        if tottf == 0:
            return numpy.zeros(self.embedding.vec_len)
        return vec/tottf




