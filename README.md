## NLPTools Repo
* This repo is a tool package for frequently used NLP tools. For more details please check [README](http://www.higgslab.com/wiki/nlptools/index.html)

* Build
    - python setup.py bdist_wheel

* Code tree
    - test
        - some test scripts
    - nlptools/utils
        some common utils
        - nlptools/utils/utils.py
            - common tools
        - nlptools/utils/dataread.py
            - table read from mysql, xls, csv, etc.
        - nlptools/utils/config.py
            - parse yaml config
        - nlptools/utils/logger.py
            - create logger
        - nlptools/utils/odsread.py
            - parse libreoffice ods file
    - nlptools/text
        some nlp tools
         -  nlptools/text/docsim.py
            * calculate similarity between vectors
         -  nlptools/text/embedding.py
            * read word2vec from several data sources redis/dynamodb/file/api
         -  nlptools/text/tokenizer.py
            * tokenizer, support jieba/mecab/ltp/corenlp/spacy/simple
         -  nlptools/text/ner.py
            * ner training and predict class, also included keyword/regex entity extraction 
         -  nlptools/text/translate.py
            * google api for translate
         -  nlptools/text/vocab.py
            * dictionary class, word/character <-> id, vec, bow 
         -  nlptools/text/topicmodel.py
            * lsi, lda model
         -  nlptools/text/acorasearch.py
            * search using acora, a keyword search engine
         -  nlptools/text/annoysearch.py
            * use annoy for fast vector based search
         -  nlptools/text/synonyms.py
            * get synonyms via word embedding
         -  nlptools/text/vectfidf.py
            * modified TF-IDF algorithm with wordvector
    - nlptools/zoo
        some universal models
         -  nlptools/zoo/comprehension/DrQA
            * Facebook's DrQA model for read comprehension
         -  nlptools/zoo/classifier
            * document classifier
         -  nlptools/zoo/demodata
            * some simple demo data
         -  nlptools/zoo/modules
            * some pytorch tools
         -  nlptools/zoo/encoders
            * universal encoder and decoder layers, can be integrated to different nlp tasks, now have traditional lstm, facebook's cnn, google's transformer 
         -  nlptools/zoo/tagging
            * tagging, language model
         -  nlptools/zoo/seq2seq
            * seq2seq models

## Version


