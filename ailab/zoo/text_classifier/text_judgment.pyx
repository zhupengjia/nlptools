#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from .text_cnn import TextCNN
from tensorflow.contrib import learn
from ailab.text import Embedding
from .data_helpers import Data_helpers
import json
import pandas as pd

class TextJudgment(object):
    def __init__(self, cfg={}):
        self.cfg = cfg
        #self.FLAGS = self.cfg['FLAGS']
        self.data_ins = Data_helpers(self.cfg)
        #self.emb_ins = Embedding(self.cfg)


    def data_process(self, positive_file, negative_file):
		# load data
        x_raw, x_text, y = self.data_ins.load_data_and_labels(positive_file, negative_file)
		
        # build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) 	
        x = np.array(list(self.vocab_processor.fit_transform(x_text)))
	
        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
	
        # Split train/test set, use cross-validation
        dev_sample_index = -1 * int(self.FLAGS['dev_sample_percentage'] * float(len(y)))
        self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print('len of train_data is:', len(self.x_train))
        print('len of dev_data is:', len(self.x_dev))
	

    def train_step(self, x_batch, y_batch):
        class_weight = sum(np.array(y_batch))/len(y_batch)
        class_weight = np.mat(class_weight)
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: self.FLAGS['dropout_keep_prob'],
            self.cnn.class_weight: class_weight}
        _, step, summaries, loss, accuracy, self.predictions, self.label = self.sess.run(
			[self.train_op, self.global_step, self.train_summary_op, self.cnn.loss, self.cnn.accuracy, self.cnn.predictions, self.cnn.label],
			feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)
	
    def dev_step(self, x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        class_weight = sum(np.array(y_batch))/len(y_batch)
        class_weight = np.mat(class_weight)
        feed_dict = {
		self.cnn.input_x: x_batch,
		self.cnn.input_y: y_batch,
		self.cnn.dropout_keep_prob: 1.0,
         self.cnn.class_weight: class_weight}
        step, summaries, loss, accuracy = self.sess.run(
			[self.global_step, self.dev_summary_op, self.cnn.loss, self.cnn.accuracy], feed_dict)
	
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    def model_path(self, path_cfg={}):
        # output directory for models and summaries
        timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        path_cfg['out_dir'] = self.out_dir
        path_cfg['checkpoint_dir'] = self.checkpoint_dir
		
        path_data = json.dumps(path_cfg)
        with open('model_path.json', 'w') as f:
            json.dump(path_data, f)
    
    def train_parameters(self):
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.cnn.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        
        # set model out put path
        self.model_path()
        print("Writing to {}\n".format(self.out_dir))
        
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", self.cnn.accuracy)
	
        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)
	
        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)
	
        # set saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS['num_checkpoints'])
    
    def train(self, restore=False):
        #set parameters
        self.FLAGS = self.cfg['FLAGS']
        #emb_ins
        self.emb_ins = Embedding(self.cfg) 

        # data process
        self.data_process(self.FLAGS['positive_data_file'], self.FLAGS['negative_data_file'])
        # train
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
				allow_soft_placement = self.FLAGS['allow_soft_placement'],
				log_device_placement = self.FLAGS['log_device_placement'])
            self.sess = tf.Session(config = session_conf)
            
            # set metrics parameters
            tf_label = tf.placeholder(dtype=tf.float32, shape=[None])
            tf_prediction = tf.placeholder(dtype=tf.float32, shape=[None])
            # define the metric and update operations
            self.tf_recall, self.tf_recall_update = tf.metrics.recall(tf_label,
                                                          tf_prediction,
                                                          name='class_metric')
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='class_metric')
            self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            
            with self.sess.as_default():
                self.cnn = TextCNN(
				sequence_length = self.x_train.shape[1],
				num_classes = self.y_train.shape[1],
				vocab_size = len(self.vocab_processor.vocabulary_),
				embedding_size = self.FLAGS['embedding_dim'],
				filter_sizes = list(map(int, self.FLAGS['filter_sizes'].split(","))),
				num_filters = self.FLAGS['num_filters'],
				l2_reg_lambda = self.FLAGS['l2_reg_lambda'])
				
                #set parameters
                self.train_parameters()
				
                #write vocabulary
                self.vocab_processor.save(os.path.join(self.out_dir, "vocab"))			
	
                #initialize all variables
                print('start initialize variables')
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(self.running_vars_initializer)
                #load pretrained embedding	
                print('start embedding')
                initW = np.random.uniform(-1, 1, (len(self.vocab_processor.vocabulary_), self.FLAGS['embedding_dim']))
                words = self.vocab_processor.vocabulary_._mapping.keys()
                for word in words:
                    idx = self.vocab_processor.vocabulary_.get(word)
                    initW[idx] = self.emb_ins[word]											
				
                self.sess.run(self.cnn.W.assign(initW))					
                #Genearte batches
                print('start generate batches')
                batches = self.data_ins.batch_iter(
					list(zip(self.x_train, self.y_train)), self.FLAGS['batch_size'], self.FLAGS['num_epochs'])
                # Training loop. For each batch:
                print('start training')
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    self.train_step(x_batch, y_batch)
                    
                    feed_dict = {tf_label:self.label, tf_prediction:self.predictions}
                    self.sess.run(self.tf_recall_update, feed_dict=feed_dict)
                    
                    current_step = tf.train.global_step(self.sess, self.global_step)
                    if current_step % self.FLAGS['evaluate_every']	== 0:
                        print("\nEvaluation:")
                        self.dev_step(self.x_dev, self.y_dev, writer = self.dev_summary_writer)
                        print("")
                    if current_step % self.FLAGS['checkpoint_every'] == 0:
                        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step = current_step)
                        print('Saved model checkpoint to {}\n'.format(path))
                score = self.sess.run(self.tf_recall)
                print('recall of train data:', score)
		
    
    def load_checkpoint(self):
		# read model path directly from cfg
        if 'model_file' in self.cfg:
            vocab_path = os.path.join(self.cfg['model_file']['out_dir'], "vocab")
            self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            
            self.checkpoint_file = tf.train.latest_checkpoint(self.cfg['model_file']['checkpoint_dir'])
        else:
			# read model path from model_path file created by train process
            if 'model_path' in self.cfg:
                with open('model_path.json', 'r') as f:
                    path_cfg = json.load(f)
                    path_cfg = json.loads(path_cfg)
                vocab_path = os.path.join(path_cfg['out_dir'], "vocab")
                self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
                self.checkpoint_file = tf.train.latest_checkpoint(path_cfg['checkpoint_dir'])

    
    def predict(self, query='', positive_file='', negative_file=''):
        self.load_checkpoint()	
        # seg sentence and map to vocabulary
        if len(positive_file) !=0 and len(negative_file) !=0:
            x_raw, x_text, y_test = self.data_ins.load_data_and_labels(positive_file, negative_file)
            y_test = np.argmax(y_test, axis=1)
        else:
            if len(query) is not 0:
                x_text = [self.data_ins.seg_sentence(query)]
                y_test = []				

        x_test = np.array(list(self.vocab_processor.transform(x_text))) #transform accept list as input

        # read model and predict
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                #load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(sess, self.checkpoint_file)

                #Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]	
				
                # Tensors to evaluate, outputs[0]输出为list
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                score = graph.get_operation_by_name("output/classify_score").outputs[0]
               				
                self.result, self.scores = sess.run([predictions, score], {input_x:x_test, dropout_keep_prob: 1.0})	
					
                if y_test is not None and len(y_test) is not 0:
                    correct_predictions = float(sum(self.result == y_test))	
                    print('Total number of test_examples: {}'.format(len(y_test)))
                    print('Accuracy: {:g}'.format(correct_predictions/float(len(y_test))))
                    
                    wrong_indexs = [i for i, j in enumerate(self.result == y_test) if j == False]
                    vocab_list = self.vocab_processor.vocabulary_._reverse_mapping
                    df_result=pd.DataFrame()
                    right_txt=[]; probability=[]; predict_class=[]
                    for idx in wrong_indexs:
                        strings=[]
                        for ID in x_test[idx]:
                            if ID !=0:
                                strings.append(vocab_list[ID])
                        wrong_text=''.join(strings)
                        right_txt.append(x_raw[idx])
                        probability.append(max(self.scores[idx]))
                        predict_class.append(self.result[idx])
#                        print('right predict text is:', x_raw[idx])
                    df_result['txt']=right_txt; df_result['class']=predict_class; df_result['score']=probability
                    df_result.sort_values(by='score', axis=0, ascending=False, inplace=True)
                    df_result = df_result.reset_index(drop=True)
                    df_result.to_excel('wrong_predict.xls', encoding='utf-8')
                                                                                                                                                                    		