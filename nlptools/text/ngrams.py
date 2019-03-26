#!/usr/bin/env python
import numpy
from .vocab import Vocab 

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Ngrams(Vocab):
    '''
        vocab support ngrams
    '''

    def __init__(self, ngrams=1, **args):
        super().__init__(**args)
        self.ngrams = ngrams


    def words2id(self, tokens, batch=False):
        '''
            tokens to token ids
            
            Input:
                - tokens: token list
                - batch: if the input sequence is a batches, default is False

            Output:
                - list of ids

        '''
        if batch:
            return numpy.asarray([self.words2id(t) for t in tokens], dtype=numpy.object) 
        ids = {}
        ids[1] = [self.word2id(t) for t in tokens]
        ids[1] = numpy.array([i for i in ids[1] if i is not None], 'int')
        for i in range(2, self.ngrams+1):
            ids[i] = []
            for j in range(len(ids[1])-i+1):
                new_token = "".join(tokens[j:j+i])
                ids[i].append(self.word2id(new_token)) 
        return ids

    @staticmethod
    def cast(vocab, ngrams=1, vocab_size=None, cached_vocab=''):
        '''
            Cast a Vocab to Ngrams vocab
            
            Input:
                - vocab: vocab instance
                - ngrams: int, default is 1
                - vocab_size: if int, will use vocab_size, default is None
                - cached_vocab: string, cached vocab file path, default is ''
        '''
        adjust_size = False if vocab_size is None else True
        vocab_size = vocab.vocab_size if vocab_size is None else vocab_size
        ngrams_vocab = Ngrams(ngrams, cached_vocab=cached_vocab, 
                vocab_size=vocab_size, outofvocab=vocab.outofvocab, 
                embedding=vocab.embedding)
        vocab_size = ngrams_vocab.vocab_size
        ngrams_vocab._word2id = vocab._word2id
        if not adjust_size:
            ngrams_vocab._id2tf = vocab._id2tf
        else:
            assert vocab_size > len(vocab._word2id)
            ngrams_vocab._id2tf = numpy.concatenate([vocab._id2tf, numpy.zeros(vocab_size-len(vocab._id2tf), "int")])
        ngrams_vocab._word_spec = vocab._word_spec
        ngrams_vocab._id_spec = vocab._id_spec
        ngrams_vocab.PAD = vocab.PAD
        ngrams_vocab.BOS = vocab.BOS
        ngrams_vocab.EOS = vocab.EOS
        ngrams_vocab.UNK = vocab.UNK
        ngrams_vocab.PAD_ID = vocab.PAD_ID
        ngrams_vocab.BOS_ID = vocab.BOS_ID
        ngrams_vocab.EOS_ID = vocab.EOS_ID
        ngrams_vocab.UNK_ID = vocab.UNK_ID
        return ngrams_vocab
         


