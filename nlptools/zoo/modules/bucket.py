#!/usr/bin/env python
import numpy as np
from nlptools.text import Vocab
from nlptools.utils import pad_sequence


def prepare_lm_data(inputs):
    ans = [(np.pad(input, (1,0), 'constant', constant_values=Vocab.BOS_ID),
           np.pad(input, (0,1), 'constant', constant_values=Vocab.EOS_ID))
            for input in inputs]
    inputs, targets = zip(*ans)
    return np.array(inputs, dtype=np.object),\
            np.array(targets, dtype=np.object)


class BucketData:
    
    def __init__(self, inputs, tags, max_words = 1000, max_seq_len=None):
        '''
        Args:
            - inputs: list of inputs, should be in the shape of (num_instance, time_step).
              However, usually the input is not an numpy nd-array, and usually is a local batch 
              of the entire dataset. 
            - tags: list of tags, same format as inputs
            - max_words: max words in one batch, used to decide the batch size
            - max_seq_len: maximum sequence length, if None then will not filter anything
        '''

        # build a list of lengths for the input data
        self.len_list = np.asarray(list(map(len, inputs)), dtype=np.integer)
        # save the sort_idx such that the original input order can be preserved. 
        self.sort_idx = np.argsort(self.len_list)[::-1]
        
        # now, let's sort the length
        self.sorted_len = self.len_list[self.sort_idx]

        self.sorted_inputs = inputs[self.sort_idx]
        self.sorted_tags = tags[self.sort_idx]

        self.max_words = max_words
        self.max_seq_len = max_seq_len
        

    def __iter__(self):
        tot_words = 0
        indexes = []
        for idx, length in enumerate(self.sorted_len):
            if self.max_seq_len is not None and length > self.max_seq_len:
                continue

            tot_words += length
            
            if tot_words > self.max_words:
                if len(indexes) > 0:
                    yield self.build_bucket(indexes)
                if length > self.max_words:
                    #single sentence length larger than max_words, then treat as solo batch
                    tot_words = 0
                    indexes = []
                    yield self.build_bucket([idx])
                else:
                    #new batch
                    indexes = [idx]
                    tot_words = length

            else:
                indexes.append(idx)

        #last batch
        if len(indexes) > 0:
            yield self.build_bucket(indexes)
            

    def build_bucket(self, indexes):
        indexes = np.array(indexes, 'int')

        #choose index
        inputs = self.sorted_inputs[indexes]
        tags = self.sorted_tags[indexes]
        lengths = self.sorted_len[indexes] 

        #join together
        inputs = pad_sequence(inputs, padding_value=Vocab.PAD_ID)
        tags = pad_sequence(tags, padding_value=Vocab.PAD_ID)

        #inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        #tags = pack_padded_sequence(tags, lengths, batch_first=True)

        return inputs, tags, lengths


def demo_data():
    
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("The dog".split(), ["DET", "NN"]),
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("Everybody read that book Everybody read that book".split(), ["NN", "V", "DET", "NN", "NN", "V", "DET", "NN"]),
        ("Everybody read that book The dog ate the apple".split(), ["NN", "V", "DET", "NN", "DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book Everybody read that book".split(), ["NN", "V", "DET", "NN", "NN", "V", "DET", "NN"])
    ]

    word_vocab = Vocab()
    tag_vocab = Vocab()
    
    inputs, tags = zip(*training_data)

    
    inputs = word_vocab(inputs, batch=True)
    tags = tag_vocab(tags, batch=True)
  
    word_vocab.reduce()
    tag_vocab.reduce()

    return inputs, tags, word_vocab, tag_vocab


if __name__ == '__main__':
    inputs, tags, _, _ = demo_data()
    print(inputs)
    print(tags)

    data = BucketData(inputs, tags, 20)
    for batch_inputs, batch_tags in data:
        print('>> batch inputs: ', batch_inputs)
        print('>> batch tags: ', batch_tags)









