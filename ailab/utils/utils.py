#!/usr/bin/env python
import os, zlib, numpy, re, pickle
from collections import Counter
from sklearn.utils import murmurhash3_32

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Some tool functions
'''


def zdump(value,filename):
    ''' 
        serialize compress variable to file using zlib and pickle
        
        input: 
            - value: python variable
            - filename: saved file path
    '''
    with open(filename,"wb",-1) as fpz:
        fpz.write(zlib.compress(pickle.dumps(value,-1),9))


#load compressed pkl file from zdump
def zload(filename):
    ''' 
        load compressed pkl file from zdump
        
        input: 
            - filename: saved file path
        
        output:
            - python variable
    '''
    with open(filename,"rb") as fpz:
        value=fpz.read()
        try:return pickle.loads(zlib.decompress(value))
        except:return pickle.loads(value)


def zdumps(value):
    '''
        serialize and compress variable to string using zlib and pickle
        
        input:
            - value: python variable
        
        output:
            - serialized string
    '''
    return zlib.compress(pickle.dumps(value,-1),9)


def zloads(value):
    ''' 
        load serialized string from zdumps
        
        input: 
            - value: serialized string
        
        output:
            - python variable
    '''
    try:return pickle.loads(zlib.decompress(value))
    except:return pickle.loads(value)


def ldumps(value):
    '''
        serialize and compress variable to string using lzo and pickle
        
        input:
            - value: python variable
        
        output:
            - serialized string
    '''
    import lzo
    return lzo.compress(pickle.dumps(value,-1),9)


def lloads(value):
    ''' 
        load serialized string from ldumps
        
        input: 
            - value: serialized string
        
        output:
            - python variable
    '''
    import lzo
    try:return pickle.loads(lzo.decompress(value))
    except:return pickle.loads(value)


def status_save(filename, status):
    '''
        save a status string to file
        
        input:
            - filename: file path
            - status: status string
    '''
    with open(filename, 'w') as f:
        f.write(str(status))


def status_check(filename):
    '''
        load a status string from file
        
        input:
            - filename: file path
        
        output:
            - status string
            - if filename not existed, return 0
    '''
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r') as f:
        return f.readlines()[0].strip()


def flat_list(l):
    '''
        flatten a 2-d list to 1-d list
        
        input:
            - l: 2-d list
        
        output:
            - 1-d list
    '''
    return [item for sublist in l for item in sublist]


def hashword(word, hashsize=16777216):
    '''
        hash the word using murmurhash3_32 to a positive int value
        
        input:
            - word: string format word
            - hashsize: maximum number, default is 16777216
        
        output:
            - int
    '''
    return murmurhash3_32(word, positive=True) % (hashsize)


def normalize(text):
    '''
        resolve different type of unicode encodings using unicodedata.normalize
        
        input:
            - text: string
        
        output:
            - string
    '''
    import unicodedata
    try:
        return unicodedata.normalize('NFD', text)
    except Exception as err:
        print(err)
        raise(err)


#rest client post
def restpost(url, data):
    '''
        rest client post using requests
        
        input:
            - url: restapi's url
            - data: python dictionary
        
        output:
            - json format post return, if failed will return None
    '''
    import requests, json
    data = requests.post(url=url, data=json.dumps(data))
    try: return data.json()
    except: return None


def envread(keys):
    '''
        use environment variables to cover original environment
        
        input:
            - keys: a key list need to read from environment
        
        output:
            - python dictionary {key:value, ...}
    '''
    cfg = {}
    for k in keys:
        if k in os.environ:
            cfg[k] = os.environ[k]
    return cfg


def distance2similarity(distance):
    '''
        Convert distance to similarity
    '''
    return 1./(1+distance)


#language detection
def lang_detect(string):
    '''
        language detection using langdetect
        
        input:
            - string
        
        output:
            - most possible language
    '''
    from langdetect import detect
    import numpy
    langs = []
    for s in re.split('\n', string):
        try: langs.append(detect(s))
        except: continue
    if len(langs) < 1: return None
    langs, langsnum = numpy.unique(langs, return_counts=True)
    return langs[numpy.argmax(langsnum)]



