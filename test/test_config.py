#!/usr/bin/env python
from nlptools.utils import Config as cfg

a = cfg('test_config.yaml')
a['x'] = [12, 13]
a.y = 14
print(a)
print(a['a'])
for k in a:
    print('-', k, a[k])
print('b' in a, 'z' in a)
print(a.a, a.c)
print(a.d.e)
print(type(a), type(a.d), type(a.d.e))


