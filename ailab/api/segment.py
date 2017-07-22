# http.py

import json
from nameko.web.handlers import http
from ailab.text import *
from ailab.utils import *

class SegmentService:
    name = "http_service"
    cfg = Config('test_jp.yaml')
    segment = Segment(cfg)

    @http('GET', '/get/<string:value>')
    def seg(self, request, value):
	return json.dumps(segmnent.seg(value))
