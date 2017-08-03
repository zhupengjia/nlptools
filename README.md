# AILab Repo

## Build
python setup.py bdist_wheel

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

## Version

### v0.0.2 [Download link](http://54.65.195.194/acn.ai/ailab/ailab/blob/c3bafaf0da809a0773e4e726ba1aead316e3c52a/dist/ailab-0.0.2-cp27-cp27mu-linux_x86_64.whl)
1. Build as wheel;
1. Adopted to iSupport iPV
