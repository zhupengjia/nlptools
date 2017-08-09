#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ailab.utils import zload
import numpy as np
import sys
from keras.layers import Input, Dense, BatchNormalization, Dropout, Embedding, Bidirectional, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.layers import add, multiply, concatenate, Lambda
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

class Supervised:
    def __init__(self, cfg, emb_ins, vocab):
        self.cfg = cfg
        self.emb_ins = emb_ins
        self.vocab = vocab
        self.learning_rate = 0.002
        self.train_epochs = 800
        self.MAX_SEQ_LEN = 50
        self.batch_size = 50
        self.embedding_len = 10000

    def build(self, n_classes):
        lstm_units = 128
        kernel_size = 5
        pooling_size = 5
        filter_size = 64
        q_input = Input(shape=(self.MAX_SEQ_LEN, ), dtype='int32')
        id2vec = np.concatenate((self.vocab.id2vec, np.random.randn(self.embedding_len-self.vocab.id2vec.shape[0], self.emb_ins.vec_len)))

        q = Embedding(self.embedding_len, self.emb_ins.vec_len, weights=[id2vec], input_length=self.MAX_SEQ_LEN, trainable=True)(q_input)
        q = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode="sum")(q)
        q = Dropout(0.2)(q)
        q = Conv1D(filter_size, kernel_size, padding='same', activation='relu', strides=1)(q)
        q = MaxPooling1D(pooling_size)(q)
        q = Dropout(0.2)(q)
        q = Conv1D(filter_size, kernel_size, padding='same', activation='relu', strides=1)(q)
        q = MaxPooling1D(pooling_size)(q)
        q = Flatten()(q)
        q = Dropout(0.2)(q)
        q = BatchNormalization()(q)
        q = Dense(n_classes, activation='softmax')(q)

        self.model = Model(inputs=q_input, outputs=q)
        #print(self.model.summary())

    def train(self, X, Y, X_valid=None, Y_valid=None):
        if X_valid is None:
            X_valid, Y_valid = X, Y
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        earlyStopping=EarlyStopping(monitor='acc', patience=50, verbose=0, mode='max')
        saveBestModel = ModelCheckpoint(self.cfg['saved_model'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.train_epochs, validation_data=(X_valid,Y_valid), callbacks=[earlyStopping, saveBestModel])
        acc = self.model.evaluate(X_valid, Y_valid)[1]
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

