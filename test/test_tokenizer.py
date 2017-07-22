#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# name:      ailab/test/test_tokenizer.py
# author:    QIAO Nan <qiaonancn@gmail.com>
# license:   GPL
# created:   2017 Jul 22
# modified:  2017 Jul 22
#

import os, sys, argparse, glob, subprocess, logging, re
from ailab.text import *
import pytest

def test_en():
    cfg={}
    cfg['LANGUAGE'] = 'en'
    o_seg = Segment(cfg)
    test_sen= "this is a test"
    assert o_seg.seg(test_sen)= {"token":[]}
    
