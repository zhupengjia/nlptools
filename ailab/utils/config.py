#!/usr/bin/env python
import os,yaml,argparse

#input can be dictionary or yaml filename
class Config(dict):
    def __init__(self, cfginput):
        config = Config.initconfig(cfginput)
        super().__init__(config)

    @staticmethod
    def initconfig(cfginput):
        if isinstance(cfginput, dict):
            config = cfginput
        elif isinstance(cfginput, argparse.Namespace):
            config = vars(cfginput)
        else:
            with open(cfginput) as f:
                config = yaml.load(f)
        for k in config:
            if isinstance(config[k], (dict, argparse.Namespace)):
                config[k] = Config(config[k])
        return config

    # dir(object)
    def __dir__(self):
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
        config = Config.initconfig(cfginput)
        for k in config:
            self[k] = config[k]

