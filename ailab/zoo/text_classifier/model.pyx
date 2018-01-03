#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from ailab.text import Embedding
from ailab.text import Segment
import json
import pandas as pd

class JudgementModel(object):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.load_checkpoint()
        self.seg_ins = Segment(self.cfg)
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False)
        
            self.sess = tf.Session(config = session_conf, graph = self.graph)
            saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
            saver.restore(self.sess, self.checkpoint_file)
        
            self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
            self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]	

            self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]
            self.score = self.graph.get_operation_by_name("output/classify_score").outputs[0]
        
		
    
    def load_checkpoint(self):
		# read model path directly from cfg
        if 'model_file' in self.cfg:
            vocab_path = os.path.join(self.cfg['model_file']['out_dir'], "vocab")
            self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            self.checkpoint_file = tf.train.latest_checkpoint(self.cfg['model_file']['checkpoint_dir'])
            self.vocab_list = self.vocab_processor.vocabulary_._reverse_mapping
        else:
			# read model path from model_path file created by train process
            if 'model_path' in self.cfg:
                with open('model_path.json', 'r') as f:
                    path_cfg = json.load(f)
                    path_cfg = json.loads(path_cfg)
                vocab_path = os.path.join(path_cfg['out_dir'], "vocab")
                self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
                self.checkpoint_file = tf.train.latest_checkpoint(path_cfg['checkpoint_dir'])
                self.vocab_list = self.vocab_processor.vocabulary_._reverse_mapping

        

    def predict(self, query=''):
        x_text = (self.seg_ins.seg_sentence(query))['tokens']
        if len(x_text)>0:
            for txt in x_text:
                if txt not in self.vocab_list:
                    x_text.remove(txt)
        x_text = [' '.join(x_text)]
        
        if len(x_text)==1 and x_text[0] =='':
            self.result = [0.5]; self.scores = [[0.5, 0.5]]
        else:
            x_test = np.array(list(self.vocab_processor.transform(x_text)))

            # Tensors to evaluate, outputs[0]输出为list
            self.result, self.scores = self.sess.run([self.predictions, self.score], {self.input_x:x_test, self.dropout_keep_prob: 1.0})	
