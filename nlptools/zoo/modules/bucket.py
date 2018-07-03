#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
from nlptools.text import Vocab


def prepare_sequence(seq, to_ix, batch_mode=False):
    '''
    Args:
        seq: the input sequence
        to_ix: the dictionary
        batch_mode: if the input sequence is are batches
    '''
    if batch_mode:
        return np.asarray([prepare_sequence(_, to_ix) for _ in seq], dtype=np.object)
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)



def prepare_lm_data(inputs):

    #print(inputs)
    #print('>>')
    def shift(input, offset, pad):
        padded = input.new_zeros(len(input) + abs(offset))
        #padded = np.zeros(len(input) + abs(offset),'int')
        if offset > 0:
            padded[offset:] = input
            padded[:offset] = pad
        else:
            padded[:len(input)] = input
            padded[len(input):] = pad
        return padded
    
    ans = [(shift(input, 1, Vocab.BOS_ID), shift(input, -1, Vocab.EOS_ID)) \
            for input in inputs]
    inputs, targets = zip(*ans)
    return np.array(inputs, dtype=np.object),\
            np.array(targets, dtype=np.object)


class BucketData:


    def __init__(self, inputs, tags, bucket_config):
        """
        Args:
            data: list of inputs, should be in the shape of (num_instance, time_step).
              However, usually the input is not an numpy nd-array, and usually is a local batch 
              of the entire dataset. 
            bucket_config: e.g., `{4: 3, 8: 2}` means for inputs of length up to 4, 
              the batch size will be 3, and for inputs of length up to 8, the batch
              size will be 2, for inputs of length more than 8, the batch size will be 1.
        """
        # build a list of lengths for the input data
        self.len_list = np.asarray(list(map(len, inputs)), dtype=np.integer)
        # save the sort_idx such that the original input order can be preserved. 
        self.sort_idx = np.argsort(self.len_list)
        
        # now, let's sort the length
        self.sorted_len = self.len_list[self.sort_idx]
        
        #self.input_data = np.asarray(data)
        #self.input_tags = np.asarray(tags)

        self.sorted_inputs = inputs[self.sort_idx]
        self.sorted_tags = tags[self.sort_idx]
        self.bucket_config = bucket_config
        
        # the actual work goes here.  
        self.build_buckets()
        
    
    def build_buckets(self):
        self.buckets = []
        i, j = 0, 0
        for max_seq_len, batch_size in self.bucket_config:
            # keep marching through the length-sorted input data
            while j < len(self.sorted_len):
                j = j + 1
                # until hit the first item longer than the `max_seq_len`
                if self.sorted_len[j - 1] > max_seq_len:
                    self.buckets.append((self.sorted_inputs[i:j - 1],
                                         self.sorted_tags[i:j - 1]))
                    i = j - 1
                    break
        # there is one more bucket that has everything left
        self.buckets.append((np.asarray(self.sorted_inputs[j - 1:]), 
                            np.asarray(self.sorted_tags[j - 1:])))
        
        
    def __iter__(self):
        for bucket_idx, bucket in enumerate(self.buckets):
            # for a bucket whose batch_size is defined in the config,
            # we use the defined batch_size
            if bucket_idx < len(self.bucket_config):
                (max_seq_len, batch_size) = self.bucket_config[bucket_idx]
                for batch in BucketData.build_batch_from_bucket(bucket, batch_size):
                    yield batch
                    
            # otherwise, we have one element each batch
            else:
                bucket_inputs, bucket_tags = bucket
                for i in range(len(bucket_inputs)):
                    yield bucket_inputs[i].view(1, -1), \
                            bucket_tags[i].view(1, -1)

                
    @staticmethod
    def build_batch_from_bucket(bucket, batch_size):
        bucket_inputs, bucket_tags = bucket
        for batch_start in range(0, len(bucket_inputs), batch_size):
            batch_inputs = bucket_inputs[batch_start : batch_start + batch_size][::-1]
            batch_tags = bucket_tags[batch_start : batch_start + batch_size][::-1]
            batch_inputs = nn.utils.rnn.pad_sequence(
                    batch_inputs, batch_first=True, padding_value=0)
            batch_tags = nn.utils.rnn.pad_sequence(
                    batch_tags, batch_first=True, padding_value=0)
            yield batch_inputs, batch_tags
        
        
    def restore(self, sequence):
        pass
    

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


    word_to_ix = {Vocab.PAD:Vocab.PAD_ID, Vocab.BOS:Vocab.BOS_ID, Vocab.EOS:Vocab.EOS_ID}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {Vocab.PAD: Vocab.PAD_ID, Vocab.BOS:Vocab.BOS_TAG, Vocab.EOS: Vocab.EOS_ID, "DET": 3, "NN": 4, "V": 5}
    inputs, tags = zip(*training_data)
    
    inputs = prepare_sequence(inputs, word_to_ix, True)
    tags = prepare_sequence(tags, tag_to_ix, True)
    
    return inputs, tags, len(word_to_ix), len(tag_to_ix)
    
    
if __name__ == '__main__':
    inputs, tags, _, _ = demo_data()
    print(inputs)
    print(tags)

    data = BucketData(inputs, tags, [(4, 3), (8, 2)])
    for batch_inputs, batch_tags in data:
        print('>> batch inputs: ', batch_inputs)
        print('>> batch tags: ', batch_tags)
