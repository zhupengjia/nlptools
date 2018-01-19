#!/usr/bin/env python
import os,yaml

class Config(dict):
    def __init__(self, yamlfile):
        if isinstance(yamlfile, dict):
            config = yamlfile
        else:
            with open(yamlfile) as f:
                config = yaml.load(f)
        for k in config:
            if isinstance(config[k], dict):
                config[k] = Config(config[k])
        super().__init__(config)

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




