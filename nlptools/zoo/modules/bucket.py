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
    
    def __init__(self, inputs, max_words = 1000, max_seq_len=None):
        '''
        Args:
            - inputs: list of data like [inputs list, tags list, ...], each one should be in the shape of (num_instance, time_step).
              However, usually the input is not an numpy nd-array, and usually is a local batch 
              of the entire dataset. 
            - max_words: max words in one batch, used to decide the batch size
            - max_seq_len: maximum sequence length, if None then will not filter anything
        '''

        # build a list of lengths for the input data
        self.len_list = np.asarray(list(map(len, inputs[0])), dtype=np.integer)
        # save the sort_idx such that the original input order can be preserved. 
        self.sort_idx = np.argsort(self.len_list)[::-1]
        
        # now, let's sort the length
        self.sorted_len = self.len_list[self.sort_idx]

        self.sorted_inputs = [ipt[self.sort_idx] for ipt in inputs]

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
        inputs = [ipt[indexes] for ipt in self.sorted_inputs]
        lengths = self.sorted_len[indexes] 

        #join together
        inputs = tuple([pad_sequence(ipt, padding_value=Vocab.PAD_ID) for ipt in inputs])

        return inputs, lengths


if __name__ == '__main__':
    inputs, tags, _, _ = demo_data()
    print(inputs)
    print(tags)

    data = BucketData(inputs, tags, 20)
    for batch_inputs, batch_tags in data:
        print('>> batch inputs: ', batch_inputs)
        print('>> batch tags: ', batch_tags)









