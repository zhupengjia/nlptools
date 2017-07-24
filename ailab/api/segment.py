# api.py

import json
from nameko.web.handlers import http
from ailab.text import *
from ailab.utils import *

class SegmentService:
    name = "segment_service"

    def __init__(self):
        cfg = Config('test_en.yaml')
        self.segment = Segment(cfg)

    @http('GET', '/get/<string:value>')
    def seg(self, request, value):
        return json.dumps(self.segment.seg(value))
