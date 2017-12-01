from data_helpers import Data_helpers
from tensorflow.contrib import learn
import numpy as np

cfg = {'LANGUAGE':'cn'}
positive_file = 'data/positive.gen'
negative_file = 'data/negative.gen'

#cfg = {'LANGUAGE': 'en'}
#positive_file = 'data/rt-polaritydata/rt-polarity.pos'
#negative_file = 'data/rt-polaritydata/rt-polarity.neg'

d = Data_helpers(cfg)
x_text, y = d.load_data_and_labels(positive_file, negative_file)

#build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

vocab=[]
for text in x_text:
	for voc in text:
		if voc not in vocab:
			vocab.append(voc)
vocab_dict = vocab_processor.vocabulary_._mapping
print('len of vocab is:', len(vocab))
print('len of voab_processor is:', len(vocab_processor.vocabulary_))
print('len of vocab_dict is:', len(vocab_dict))
print('type of vocab_dict is:', type(vocab_dict))
print(vocab_dict.keys())
print(len(x_text))
print(x_text[:6])
print(len(y))
