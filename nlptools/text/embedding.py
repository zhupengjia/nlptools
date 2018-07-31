#!/usr/bin/env python
# -*- coding: utf-8 -*-
#word2vec

import time, base64, os
import numpy as np
from scipy.spatial.distance import cosine
from ..utils import zload, zdump, restpost


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Embedding_Base(object):
    '''
        Parent class for other embedding classes to read the word vectors, please don't use this class directly

        Input:
            - dim: int, vector length, default is 300
            - data_type: type of saved vector, default is float64
            - cached_data: the cache path in local, default is ''
            - additional_dim: 0 for random, 1 for random but with additional dim, default is 0
            - base64: bool, if True, will return BASE64 instead of vector
    '''
    def __init__(self, dim=300, data_type='float64', cached_data='', additional_dim=0, base64=False):
        if additional_dim:
            self.dim = int(dim) + 1
        else:
            self.dim = int(dim)
        self.cached_data = cached_data
        self.data_type =data_type
        self.additional_dim = additional_dim
        self.base64 = base64
        self.__get_cached_vec()
   

    def distance(self, word1, word2):
        '''
            Calculate the cosine distance for two words

            Input:
                - word1: string
                - word2: string
        '''
        vec1 = self.__getitem__(word1)
        vec2 = self.__getitem__(word2)
        return cosine(vec1, vec2)


    def __len__(self):
        if self.additional_dim:
            return self.dim+1
        else:
            return self.dim


    def __get_cached_vec(self):
        if os.path.exists(self.cached_data):
            self.cached_vec = zload(self.cached_data)
        else:
            self.cached_vec = {}
   
    
    def _postdeal(self, v = None, returnbase64 = False):
        if self.additional_dim:
            if v:
                v = np.concatenate((v, np.zeros(1)))
            else:
                v = np.concatenate((np.random.randn(self.dim - 1), np.ones(1)))
        else:
            if v is None:
                v = np.random.randn(self.dim)
        if returnbase64:
            v = base64.b64encode(v.tostring()).decode()
        return v
    

    def save(self):
        '''
            Save the word vectors in memory to *cached_data*
        '''
        if len(self.cached_data) > 0:
            zdump(self.cached_vec, self.cached_data)


class Embedding_File(Embedding_Base):
    '''
        Read the word vectors from local files

        Input:
            - w2v_word2idx: pickle filepath for word-idx mapping
            - w2v_idx2vec: numpy dump for idx-vector mapping
            - any parameters mentioned in Embedding_Base

        Usage:
            - emb_ins[word]: return the vector or BASE64 format vector
            - word in emb_ins: check if word existed in file
    '''
    def __init__(self, w2v_word2idx, w2v_idx2vec, **args):
        Embedding_Base.__init__(self, **args)
        self.word2idx = None
        self.w2v_idx2vec = w2v_idx2vec
        self.w2v_word2idx = w2v_word2idx

    def _load_vec(self):
        if self.word2idx is None:
            self.word2idx = zload(self.w2v_word2idx)
            self.idx2vec = np.load(self.w2v_idx2vec).astype('float')

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        self._load_vec()
        v = self.idx2vec[self.word2idx[word]] if word in self.word2idx else None
        v = self._postdeal(v, self.base64)
        self.cached_vec[word] = v
        #print(v)
        return v

    def __contains__(self, word):
        self._load_vec()
        return word in self.word2idx


class Embedding_Redis(Embedding_Base):
    '''
        Read the word vectors from redis

        Input:
            - redis_host: redis host
            - redis port: redis port
            - redis_db: database in redis
            - any parameters mentioned in Embedding_Base

        Usage:
            - emb_ins[word]: return the vector or BASE64 format vector
            - word in emb_ins: check if word existed in database
    '''
    def __init__(self, redis_host, redis_port, redis_db, **args):
        import redis
        Embedding_Base.__init__(self, **args)
        self.redis_ins = redis.Redis(connection_pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=redis_db))

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = self.redis_ins.get(word)
        if v is None:
            v = self._postdeal(v, self.base64)
        else:
            if not self.base64:
                v = self._postdeal(np.fromstring(base64.b64decode(v)), self.base64)
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        v = self.redis_ins.get(word)
        return v is not None


class Embedding_Random(Embedding_Base):
    '''
        Randomly generate the vector for word

        Input:
            - any parameters mentioned in Embedding_Base

        Usage:
            - emb_ins[word]: return the vector or BASE64 format vector
            - word in emb_ins: check if word existed in cache
    '''
    def __init__(self, **args):
        Embedding_Base.__init__(self, **args)
        self.dim = int(self.dim)

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = self._postdeal(None, self.base64)
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.cached_vec


class Embedding_Dynamodb(Embedding_Base):
    '''
        Read the word vectors from aws dynamodb, please make sure you have related credential files for enviroments.

        Input:
            - dynamodb: dynamodb database name
            - all keys mentioned in Embedding_Base

        Usage:
            - emb_ins[word]: return the vector or BASE64 format vector
            - word in emb_ins: check if word existed in database
    '''
    def __init__(self, dynamodb, **args):
        import boto3
        Embedding_Base.__init__(self, **args)
        self.client = boto3.resource(dynamodb)
        self.table = self.client.Table(dynamodb)

    def __getitem__(self, word):
        import boto3
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = self.table.get_item(Key={"word":word})
        if not "Item" in v:
            v = self._postdeal(None, self.base64)
        else:
            vector_binary = v['Item']['vector']
            if isinstance(vector_binary, boto3.dynamodb.types.Binary):
                vector_binary = vector_binary.value
            if self.base64:
                v = vector_binary
            else:
                vector = np.fromstring(base64.b64decode(vector_binary), dtype=self.data_type)
                v = self._postdeal(vector)
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        v = self.table.get_item(Key={"word":word})
        return "Item" in v


class Embedding_Rest(Embedding_Base):
    '''
        Read the word vectors from restapi

        Input:
            - embedding_restapi: string, restapi url 
            - data_type: the type of vector used in restapi, float64, float32, float
         
        Usage:
            - emb_ins[word]: return the vector or BASE64 format vector
            - word in emb_ins: check if word existed in database
    '''
    def __init__(self, embedding_restapi, **args):
        Embedding_Base.__init__(self, **args)
        self.rest_url = embedding_restapi
    
    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        vector = restpost(self.rest_url, {'text':word})
        if not base64:
            vector = np.fromstring(base64.b64decode(vector), dtype=self.data_type)
            vector = self._postdeal(vector)
        self.cached_vec[word] = vector
        return vector

    def __contains(self, word):
        return True


class Embedding(object):
    '''
        Read the word vectors from different database sources

        Input:
            - automatically choose embedding from parameters, if exists:
                1. *w2v_word2idx* *w2v_idx2vec* read the wordvec from file
                2. *dynamodb* read the wordvec from amazon's dynamodb
                3. *redis_host* read from redis
                4. *embedding_restapi* read from restapi
                5. default: random generated
    '''
    def __new__(cls, **args):
        if 'w2v_word2idx' in args and 'w2v_idx2vec' in args:
            return Embedding_File(**args)
        elif 'dynamodb' in args:
            return Embedding_Dynamodb(**args)
        elif 'redis_host' in args:
            return Embedding_Redis(**args)
        elif 'embedding_restapi' in args:
            return Embedding_Rest(**args)
        else:
            return Embedding_Random(**args)

            


