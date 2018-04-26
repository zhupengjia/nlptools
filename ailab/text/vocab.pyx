#!/usr/bin/env python
import numpy, os, random
import unicodedata
from .tokenizer import Segment, Segment_Char
from ..utils import zload, zdump, hashword, normalize, flat_list

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

# get TF of vocabs and vectors
class Vocab(object):
    '''
        Vocab dictionary class, also support to accumulate term frequency, return embedding matrix, sentence to id

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - cached_vocab: string, cached vocab file path, default is ''
                    - vocab_size: int, the dictionary size, default is 2**24. If the number<30, then the size is 2**vocab_size, else the size is *vocab_size*.
                    - ngrams: int, ngrams for sentence2id function, default is 1
                    - outofvocab: string, the behavior of outofvocab token, default is 'unk'. Two options: 'unk' and 'random'. 'unk' will fill with 'unk', 'random' will fill with a random token
                    - hashgenerate: bool, the token id generate from hash or accumulating. Default is False
            - seg_ins: instance of ailab.text.tokenizer.segment
            - emb_ins: instance of ailab.text.embedding
            - forceinit: bool, if true will always init a new vocab. Default is False  

        Some special operation:
            - __add__: join several ailab.text.vocab together
            - __call__: get ids for sentence list. Input is [sentence, ...], or [token_list, ...]
            - __getitem__: get id for word
            - __contains__: checkout if word or id in vocab
    '''
    def __init__(self, cfg=None, seg_ins=None, emb_ins=None, forceinit=False):
        self.cfg = {'cached_vocab': '', 'vocab_size': 2**24, 'ngrams':1, 'outofvocab':'unk', 'hashgenerate':0}
        if cfg is not None:
            for k in cfg: self.cfg[k] = cfg[k]
        if isinstance(self.cfg['ngrams'], int):
            self.cfg['ngrams'] = list(range(1, self.cfg['ngrams']+1))
        self.seg_ins = seg_ins
        self.seg_ins_emb = seg_ins is None # if tokenizer embedded in Vocab or as a parameter input
        self.seg_char = Segment_Char(cfg)
        self.emb_ins = emb_ins
        self.vocab_size = int(self.cfg['vocab_size'])
        if self.vocab_size < 30:
            self.vocab_size = 2**int(self.cfg['vocab_size'])
        self.__get_cached_vocab(forceinit)

    def __del__(self):
        del self._id2word, self._word2id, self._id2tf, self._id_ngrams
        if self.seg_ins_emb: 
            del self.seg_ins

    def addBE(self):
        '''
            Add 4 special characters to dictionary, '<pad>', '<eos>', '<bos>','<unk>'
        '''
        self._word_spec = ['<pad>', '<eos>', '<bos>','<unk>']
        self._id_spec = [self.word2id(w) for w in self._word_spec]
        self.PAD, self.EOS, self.BOS, self.UNK = tuple(self._word_spec)
        self._id_PAD, self._id_EOS, self._id_BOS, self._id_UNK = tuple(self._id_spec)

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
            ids = idlist
        else:
            if isinstance(wordlist[0], list):
                wordlist = flat_list(wordlist)
            if isinstance(wordlist, str) or isinstance(wordlist[0], str):
                ids = self.sentence2id(wordlist, update=False)
            elif isinstance(wordlist[0], int):
                #if wordlist is an idlist
                ids = wordlist
            else:
                raise('docbow input is not supported!  your input is:' + str(wordlist))
        if isinstance(ids[0], list):
            ids = flat_list(ids)
        tfs = [self._id2tf[i] for i in ids]
        return list(zip(ids, tfs))

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
                self.vocab_size = len(self._id2tf)
            except Exception as err:
                print('warning!!! cached vocab read failed!!! ' + err)
        if ifinit:
            self._id2word = {}
            self._word2id = {}
            self._id2tf = numpy.zeros(self.vocab_size, 'int')
            self._vocab_max = -1
            self._id_ngrams = {}
        if any([n>1 for n in self.cfg['ngrams']]):
            self.addBE()

    def save(self):
        '''
            Save the vocab dictionary to *cached_vocab*
        '''
        if len(self.cfg['cached_vocab']) > 0:
            zdump((self._id2word, self._word2id, self._id2tf, self._id_ngrams), self.cfg['cached_vocab'])


    def accumword(self, word, fulfill=True, tfaccum = True):
        '''
            Add the word to dictionary

            Input: 
                - word: string
                - fulfill: bool, check if the word used to fulfill the dictionary, default is True
                - tfaccum: bool, check if accumulate term count. Default is True
            
            Output:
                - wordid: int    
        '''
        if word is None: return None
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
                    if self.cfg['outofvocab']=='random':
                        wordid = random.randint(0, self.vocab_size-1)
                    else:
                        wordid = self._id_UNK
                    self._id2tf[wordid] += 1
            self._word2id[word] = wordid
            return wordid
        else:
            return None


    @property
    def Nwords(self):
        '''
            Property of the class

            Output:
                - number of words
        '''
        return len(self._id2word)


    def __len__(self):
        return len(self._id2word)

    def reduce_vocab(self, vocab_size=None, reorder=False):
        '''
            reduce vocab size by tf.
            
            Input: 
                - vocab_size: int, the target vocab_size, if None will reduce to the current number of words. Default is None
                - reorder: bool, check if the dictionary reorder by tf. Default is False. Parameter only usable when vocab_size is None. If the vocab_size is not None, the dictionary will always be reordered
        '''
        if vocab_size is None:
            self.vocab_size = self._vocab_max + 1
            self._id2tf = self._id2tf[:self.vocab_size]
            vocab_size = self.vocab_size
            if not reorder:
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
    

    #merge another vocab
    def __add__(self, other):
        id_mapping = {}
        for w in other._word2id:
            _id_other = other._word2id[w]
            if w in self._word2id:
                _id = self._word2id[w]
                self._id2tf[_id] += other._id2tf[_id_other]
            else:
                _id = self.add_word(w)
                self._id2tf[_id] = other._id2tf[_id_other]
            id_mapping[_id_other] = _id
        for _id_other in other._id_ngrams:
            if _id_other in self._id_ngrams:
                continue
            self._id_ngrams[id_mapping[_id_other]] = [id_mapping[i] for i in other._id_ngrams[_id_other]]
        return self

    def __call__(self, sentences):
        '''
            call function, convert sentences to id

            Input:
                - sentences: list of sentence or list of tokens

            Output:
                - list of ids: [[token_id1, ...], ...]
        '''
        ids = []
        for sentence in sentences:
            ids.append(self.sentence2id(sentence))
        return ids
  
    def __getitem__(self, key):
        '''
            get word from id
        '''
        return self.word2id(key, fulfill=False)

    def __contains__(self, key):
        '''
            check if word or id in vocab

            Input:
                - key: if key is int , then check if id in vocab, otherwise check if word in vocab
        '''
        if isinstance(key, int):
            return key in self._id2word
        return key in self._word2id

    def add_word(self, word):
        '''
            add word to vocab, same as accumword(word, fulfill=True, tfaccum=True)

            Input:
                - word: string

            Output:
                - wordid: int
        '''
        return self.accumword(word)


    def word2id(self, word, fulfill=True):
        '''
            convert word to id but not accumulate tf, same as accumword(word, fulfill, tfaccum=False)

            Input:
                - word: string
                - fulfill: bool, check if the word used to fulfill the dictionary, default is True

            Output:
                - wordid: int
        '''
        return self.accumword(word, fulfill, False)

    def id2word(self, i):
        '''
            convert id to word.

            Input:
                - i: wordid

            Output:
                - string. If id not in vocab, will return None
        '''
        if i in self._id2word:
            return self._id2word[i]
        return None

    def sentence2id(self, sentence, ngrams=None, useBE=True, update=True, charlevel=False, charngram=False, remove_stopwords=True, flatresult=True):
        '''
            sentence to  token ids
            
            Input:
                - sentence: string or token list
                - ngrams: int, the returned token ids will also include multi-grams ids, if None will use the config of 'ngrams'. Default is None
                - useBE: bool, True will add <BOS> and <EOS> for multi-grams ids. False will not. Default is True
                - update: bool, check if use tf accumulate, default is True
                - charlevel: bool, check if also include char level ids in token ids, default is False
                - charngram: bool, check if ngrams parameter also affect char level ids, default is False
                - remove_stopwords: bool, check if remove stopwords, default is True
                - flatresult: bool, check if flat the final result. Please check the output. Default is True

            Output:
                - if flatresult is True, then return a list of ids
                - if flatresult is False, then return a dictionary, key is the n of n-grams and 'char', like 1,2,3,..,,char , value is the word id list.

        '''
        if isinstance(sentence, str):
            if self.seg_ins is None:
                self.seg_ins = Segment(self.cfg)
            sentence_seg = self.seg_ins.seg(sentence, remove_stopwords=remove_stopwords)['tokens']
            if charlevel and charngram:
                sentence_seg += self.seg_char.seg(sentence, remove_stopwords=remove_stopwords)['tokens']
        elif sentence == None:
            if flatresult: return []
            else: return {}
        else:
            sentence_seg = sentence

        if ngrams is None:
            ngrams = self.cfg['ngrams']
        elif isinstance(ngrams, int):
            ngrams = list(range(1, ngrams+1))
            
        if update:
            func_add_word = self.add_word
        else:
            func_add_word = self.word2id
        
        ids = {}
        ids[1] = [func_add_word(t) for t in sentence_seg]
        ids[1] = [i for i in ids[1] if i is not None]
        if len(ids[1]) < 1:
            if flatresult: return []
            else: return {}

        ngrams2 = [n for n in ngrams if n > 1]
        #ngrams
        if len(ngrams2) > 0:
            if useBE:
                ids_BE = [self._id_BOS] + ids[1] + [self._id_EOS]
                words_BE = [self.BOS] + sentence_seg + [self.EOS]
            else:
                ids_BE = ids[1]
                words_BE = sentence_seg
        for n in ngrams2:
            if len(ids_BE) < n: break
            ids_gram_tuple = [ids_BE[i:i+n] for i in range(len(ids_BE)-n+1)]
            ids_gram_word = [''.join(words_BE[i:i+n]) for i in range(len(words_BE)-n+1)]
            ids[n] = [func_add_word(x) for x in ids_gram_word]
            for d in zip(ids[n], ids_gram_tuple):
                if not d[0] in self._id_ngrams:
                    self._id_ngrams[d[0]] = d[1]
        #charlevel
        if charlevel:
            ngrams.append('char')
        if charlevel and not charngram:
            sentence_char = self.seg_char.seg(sentence, remove_stopwords=remove_stopwords)['tokens']
            ids['char'] = [func_add_word(t) for t in sentence_char]
            ids['char'] = [i for i in ids['char'] if i is not None]
        ids = {n:ids[n] for n in ngrams if n in ids}
        if flatresult:
            return flat_list(ids.values())
        else:
            return ids

    def id2sentence(self, ids):
        '''
            convert id back to sentence

            Input: 
                - ids: list of ids
        '''
        return ' '.join([self._id2word[i] for i in ids])

    
    def get_id2vec(self):
        '''
            Get all word vectors for all word_ids, Used to cache word vectors, combine with save function 
        '''
        if self.emb_ins is None:
            return None
        for i in self._id2word:
            self.id2vec(i)
    
    def dense_vectors(self):
        '''
            return a numpy array of word vectors. The index of array is the word_ids
        '''
        self.get_id2vec()
        vectors = numpy.zeros((self._vocab_max+1, self.emb_ins.vec_len), 'float')
        for k in self._id2word:
            if k == self._id_PAD:
                vectors[k] = 0
            else:
                vectors[k] = self.id2vec(k)
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
        if self.emb_ins is None:
            return None
        vec = numpy.zeros((len(sentence_id), self.emb_ins.vec_len), 'float')
        for i,sid in enumerate(sentence_id):
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
        if word_id in self._id_ngrams:
            return numpy.sum([self.emb_ins[self._id2word[ii]] for ii in self._id_ngrams[word_id]], axis=0)
        else:
            return self.emb_ins[self._id2word[word_id]]


    def ave_vec(self, sentence_id):
        '''
            get tf weighted average word vector of token list

            Input:
                - sentence_id: token_id list
            
            Output:
                - 1d numpy array
        '''
        vec = numpy.zeros(self.emb_ins.vec_len, 'float')
        tottf = 0
        for i, sid in enumerate(sentence_id):
            w = numpy.log(self.Nwords/(self._id2tf[sid]))
            vec += self.emb_ins[self._id2word[sid]]*w
            tottf += w
        if tottf == 0:
            return numpy.zeros(self.emb_ins.vec_len)
        return vec/tottf




