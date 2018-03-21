## AILab Repo
* This repo is a tool package for frequently used NLP tools. For more details please check [README](http://n2.c3.acnailab.com/code/ailab/index.html)

* Build
    - python setup.py bdist_wheel

* Code tree
    - test
        - pytest codes
    - ailab/utils
        some common utils
        - ailab/utils/utils.py
            - common tools
        - ailab/utils/qnaread.py
            - table read from mysql, xls, csv, etc.
        - ailab/utils/config.py
            - parse yaml config
        -  ailab/utils/logger.py
            - create logger
        - ailab/text
            - some nlp tools
         -  ailab/text/docsim.pyx
            - calculate distance between vectors
         -  ailab/text/embedding.pyx
            - read word2vec from redis/dynamodb/file/api
         -  ailab/text/tokenizer.pyx
            - tokenizer, support jieba/mecab/ltp/corenlp/spacy/simple
         -  ailab/text/ner.pyx
            - ner training and predict class, also included keyword/regex entity extraction 
         -  ailab/text/translate.pyx
            - google api to translate
         -  ailab/text/vocab.pyx
            - dictionary class, word/character <-> id, vec, bow 
         -  ailab/text/topicmodel.pyx
            - lsi, lda model
         -  ailab/text/acorasearch.pyx
            - search using acora, a keyword search engine
         -  ailab/text/annoysearch.pyx
            - use annoy for fast vector based search
    - ailab/zoo
        some models

## Version


