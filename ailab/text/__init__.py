#!/usr/bin/env python
import pyximport
pyximport.install()

from .tokenizer import Segment

__all__ = ["Segment"]
