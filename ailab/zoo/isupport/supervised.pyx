#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ailab.utils import zload
from .models import v1
import numpy as np
import sys
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

class Supervised:
    def __init__(self, cfg, emb_ins, vocab):
        self.cfg = cfg
        self.emb_ins = emb_ins
        self.vocab = vocab
        self.learning_rate = self.cfg['learning_rate']
        self.train_epochs = self.cfg['train_epochs']
        self.batch_size = self.cfg['batch_size']
        self.MAX_SEQ_LEN = self.cfg['max_seq_len']
        self.tf_session()
    
    def tf_session(self):
        config = tf.ConfigProto(intra_op_parallelism_threads=self.cfg['num_cores'],\
                inter_op_parallelism_threads=self.cfg['num_cores'], allow_soft_placement=True,\
                device_count = {'CPU' : self.cfg['num_cpu'], 'GPU' : self.cfg['num_gpu']})
        session = tf.Session(config=config)
        K.set_session(session)

    def build(self, n_classes):
        self.model = v1(n_classes, \
                embedding_len = self.emb_ins.vec_len, \
                embedding_matrix = self.vocab._id2vec, \
                max_seq_len = self.MAX_SEQ_LEN \
                )

    def train(self, X, Y, X_valid=None, Y_valid=None):
        if X_valid is None:
            X_valid, Y_valid = X, Y
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        earlyStopping=EarlyStopping(monitor='acc', patience=self.cfg['earlystop_patience'], verbose=0, mode='max')
        saveBestModel = ModelCheckpoint(self.cfg['saved_model'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.train_epochs, validation_data=(X_valid,Y_valid), callbacks=[earlyStopping, saveBestModel])
        acc = self.model.evaluate(X_valid, Y_valid)[1]
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        return acc

    def eval(self, X, Y):
        return self.model.evaluate(X, Y)[1]
    
    def pred(self, X):
        return self.model.predict(X)
       
    def save(self):
        self.model.save(self.cfg['saved_model'])
    
    def load(self):
        self.model = load_model(self.cfg['saved_model'])
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def reset(self):
        self.model.reset_states()

