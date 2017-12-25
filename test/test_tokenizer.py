#!/usr/bin/env python3
from ailab.text.tokenizer import Segment_Simple
from ailab.utils import zload
import sys

cfg = {'TOKENIZER':'simple'}
s = Segment_Simple(cfg)
#print(s.seg(sys.argv[1]))

print(s.seg('浙江在线杭州4月25日讯（记者施宇翔 通讯员 方英）毒贩很“时髦”，用微信交易毒品. N is batch size; D_in is input dimension;'))

