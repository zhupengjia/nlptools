#!/usr/bin/env python
import pyximport
pyximport.install()

from .tokenizer import Segment
from .embedding import Embedding
from .vocab import Vocab
from .docsim import DocSim
from .translate import Translate
from .vectfidf import VecTFIDF
from .synonyms import Synonyms
from .annoysearch import AnnoySearch
from .acorasearch import AcoraSearch
from .ner import NER

__all__ = ["Segment", 'Embedding', 'Vocab', 'DocSim', 'VecTFIDF', 'Synonyms', 'AnnoySearch', 'AcoraSearch', 'Translate', 'NER']
