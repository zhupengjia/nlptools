#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import random
import itertools
from collections import Counter
from ailab.text import Segment

class Data_helpers(object):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.seg_ins = Segment(self.cfg)		

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
	
    def seg_sentence(self, sent):
        sentence_seg = (self.seg_ins.seg_sentence(sent))['tokens']
        sentence = ' '.join(sentence_seg)
        return sentence

    def data_concat(self, data, length):
        step = int(len(data)/length)
        random.shuffle(data)
        data_new = []
        for i in range(step):
            elements = []
            for j in range(length):
                elements.append(data[i+j*step])
            data_new.append(','.join(elements))
        return data_new

    def data_new(self, positive_data_file, negative_data_file, length):
        #read data
        positive_examples = list(open(positive_data_file, 'r').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, 'r').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        #concat data
        new_positive = self.data_concat(positive_examples, length)
        new_negative = self.data_concat(negative_examples, length)

        #write to file
        for item in new_positive:
            with open(positive_data_file, 'a') as f:
                f.write(item+'\n')
        for negative in new_negative:
            with open(negative_data_file, 'a') as f:
                f.write(negative+'\n')
    
    def load_data_and_labels(self, positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
    
        # Split by words
        x_raw = positive_examples + negative_examples
#       print('len of x_text is:', len(x_text))
        if self.cfg['LANGUAGE'] == 'en':
            x_text = [self.clean_str(sent) for sent in x_raw]
        print('first 6 sentences is:',x_raw[:6])
        x_text = [self.seg_sentence(sent) for sent in x_raw]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_raw, x_text, y]


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
        # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
