#!/usr/bin/env python3
from ailab.text.tokenizer import Segment_Rest

cfg = {'TOKENIZER':'http://127.0.0.1:8000/api/tokenize/'}
t = Segment_Rest(cfg)

#text = 'Change syntax themes, default project pages, and more in preferences.\n hello world'
text = "今天天气好不错啊"

print(t.seg(text))



