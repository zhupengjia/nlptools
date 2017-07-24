## Code tree
### test
pytest codes
### ailab/utils
some common utils
#### ailab/utils/utils.py
common tools
#### ailab/utils/qnaread.py
table read from mysql, xls, csv, etc.
#### ailab/utils/config.py
read yaml config
#### ailab/utils/logger.py
create logger
### ailab/text
some nlp tools
#### ailab/text/docsim.pyx
calculate distance between vectors
#### ailab/text/embedding.pyx
read word2vec from redis/dynamodb/file
#### ailab/text/tokenizer.pyx
tokenizer, support cn/jp/en/yue
#### ailab/text/translate.pyx
google api to translate
#### ailab/text/vocab.pyx
class to convert word/charactor to id, vector, build dictionary, etc
#### ailab/text/topicmodel.pyx
lsi, lda model

