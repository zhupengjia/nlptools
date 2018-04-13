#!/usr/bin/env python
import os,argparse,yaml

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

#input can be dictionary or yaml filename
class Config(dict):
    '''
        Configuration file read from yaml, inherit from python dictionary

        Input:
            - cfginput: yaml filepath, python dictionary, argparse.namespace

        Usage:
            - cfg.a.b
            - cfg['a']['b']

    '''
    def __init__(self, cfginput):
        config = Config._initconfig(cfginput)
        super().__init__(config)

    @staticmethod
    def _initconfig(cfginput):
        if isinstance(cfginput, dict):
            config = cfginput
        elif isinstance(cfginput, argparse.Namespace):
            config = vars(cfginput)
        else:
            with open(cfginput, encoding='utf-8') as f:
                print('==========encoding utf-8')
                config = yaml.load(f)
            
        for k in config:
            if isinstance(config[k], (dict, argparse.Namespace)):
                config[k] = Config(config[k])
        return config

    # dir(object)
    def __dir__(self):
        '''
             overide dir(object)
        '''
        return tuple(self)


    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("config has no attribute '{}'".format(key))


    def __setattr__(self, key, item):
        self[key] = item


    def __delattr__(self, key):
        del self[key]


    def update(self, cfginput):
        '''
            update configurations from other configurations

            input:
                - cfginput: Config object for python dictionary
        '''
        config = Config._initconfig(cfginput)
        for k in config:
            self[k] = config[k]

