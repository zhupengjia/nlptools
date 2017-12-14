#!/usr/bin/env python
import os,yaml

class Config:
    def __init__(self, yamlfile):
        with open(yamlfile) as f:
            self.config = yaml.load(f)

    def __setitem__(self, key, item):
        self.config[key] = item

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        for k in self.config.keys():
            yield k

    def __contains__(self, key):
        return key in self.config


