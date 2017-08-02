#!/usr/bin/env python
# -*- coding: utf-8 -*-
#word2vec

import time, redis, boto3, base64, os
import numpy as np
from scipy.spatial.distance import cosine
from ailab.utils import zload, zdump


class Embedding_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.__get_cached_vec()
        self.vec_len = cfg['vec_len'] + 1
        if 'vec_type' in cfg:
            self.vec_type = cfg['vec_type']
        else:
            self.vec_type = 'float64'

    def distance(self, word1, word2):
        vec1 = self.__getitem__(word1)
        vec2 = self.__getitem__(word2)
        return cosine(vec1, vec2)

    def __get_cached_vec(self):
        if os.path.exists(self.cfg['cached_w2v']):
            self.cached_vec = zload(self.cfg['cached_w2v'])
        else:
            self.cached_vec = {}
    
    def save(self):
        zdump(self.cached_vec, self.cfg['cached_w2v'])


class Embedding_File(Embedding_Base):
    def __init__(self, cfg):
        Embedding_Base.__init__(self, cfg)
        self.word2idx = None

    def __load_vec(self):
        if self.word2idx is None:
            self.word2idx = zload(self.cfg['w2v_word2idx'])
            self.idx2vec = np.load(self.cfg['w2v_idx2vec'])

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        self.__load_vec()
        if word in self.word2idx:
            v = np.concatenate((self.idx2vec[self.word2idx[word]],np.zeros(1))).astype('float32')
        else:
            v = np.concatenate((np.random.randn(self.vec_len - 1),np.ones(1))).astype('float32')
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        self.__load_vec()
        return word in self.word2idx


class Embedding_Redis(Embedding_Base):
    def __init__(self, cfg):
        Embedding_Base.__init__(self, cfg)
        self.redis_ins = redis.Redis(connection_pool = redis.ConnectionPool(host=cfg["redis_host"], port=cfg["redis_port"], db=cfg["redis_db"]))

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = self.redis_ins.get(word)
        if v is not None:
            v = np.concatenate((np.fromstring(v),np.zeros(1))).astype('float32')
        else:
            v = np.concatenate((np.random.randn(self.vec_len - 1),np.ones(1))).astype('float32')
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        v = self.redis_ins.get(word)
        return v is not None

class Embedding_Random(Embedding_Base):
    def __init__(self, cfg):
        Embedding_Base.__init__(self, cfg)

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = np.random.randn(self.vec_len -1).astype('float32')
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.cached_vec

class Embedding_Dynamodb(Embedding_Base):
    def __init__(self, cfg):
        Embedding_Base.__init__(self, cfg)
        self.client = boto3.resource('dynamodb')
        self.table = self.client.Table(cfg['dynamodb'])

    def __getitem__(self, word):
        if word in self.cached_vec:
            return self.cached_vec[word]
        v = self.table.get_item(Key={"word":word})
        if "Item" in v:
            vector_binary = v['Item']['vector']
            if isinstance(vector_binary, boto3.dynamodb.types.Binary):
                vector_binary = vector_binary.value
            vector = np.fromstring(base64.b64decode(vector_binary), dtype=self.vec_type)
            v = np.concatenate((vector, np.zeros(1))).astype('float32')
        else:
            v = np.concatenate((np.random.randn(self.vec_len - 1),np.ones(1))).astype('float32')
        self.cached_vec[word] = v
        return v

    def __contains__(self, word):
        v = self.table.get_item(Key={"word":word})
        return "Item" in v

class Embedding(object):
    def __new__(cls, cfg):
        if 'w2v_word2idx' in cfg and 'w2v_idx2vec' in cfg:
            return Embedding_File(cfg)
        elif 'dynamodb' in cfg:
            return Embedding_Dynamodb(cfg)
        elif 'redis_host' in cfg:
            return Embedding_Redis(cfg)
        else:
            return Embedding_Random(cfg)

            


