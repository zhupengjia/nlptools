## NLPTools Repo
* This repo is a tool package for frequently used NLP tools. For more details please check [README](http://n2.c3.acnailab.com/code/ailab/index.html)

* Build
    - python setup.py bdist_wheel

* Code tree
    - test
        - pytest codes
    - nlptools/utils
        some common utils
        - nlptools/utils/utils.py
            - common tools
        - nlptools/utils/qnaread.py
            - table read from mysql, xls, csv, etc.
        - nlptools/utils/config.py
            - parse yaml config
        -  nlptools/utils/logger.py
            - create logger
    - nlptools/text
        some nlp tools
         -  nlptools/text/docsim.pyx
            * calculate distance between vectors
         -  nlptools/text/embedding.pyx
            * read word2vec from redis/dynamodb/file/api
         -  nlptools/text/tokenizer.pyx
            * tokenizer, support jieba/mecab/ltp/corenlp/spacy/simple
         -  nlptools/text/ner.pyx
            * ner training and predict class, also included keyword/regex entity extraction 
         -  nlptools/text/translate.pyx
            * google api for translate
         -  nlptools/text/vocab.pyx
            * dictionary class, word/character <-> id, vec, bow 
         -  nlptools/text/topicmodel.pyx
            * lsi, lda model
         -  nlptools/text/acorasearch.pyx
            * search using acora, a keyword search engine
         -  nlptools/text/annoysearch.pyx
            * use annoy for fast vector based search
         -  nlptools/text/synonyms.pyx
            * get synonyms via word embedding
    - nlptools/zoo
        some models

## Version


