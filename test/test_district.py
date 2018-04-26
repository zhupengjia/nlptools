#!/usr/bin/env python
from ailab.utils.district_cn import District_CN

d = District_CN("/home/pzhu/data/district_cn")
print(d.get_nodes('金华'))
print(d.get_nodeids('金华'))


