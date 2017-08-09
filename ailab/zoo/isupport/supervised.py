#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from keras.layers import Input, Dense, BatchNormalization, Dropout, Embedding, Bidirectional, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.layers import add, multiply, concatenate, Lambda
from keras.models import Model, load_model
import tensorflow as tf

def v1(n_classes, embedding_len, embedding_matrix, max_seq_len=50, embedding_size=10000, lstm_units=128, kernel_size=5, pooling_size=5, filter_size=64, dropout=0.2):
    q_input = Input(shape=(max_seq_len, ), dtype='int32')
    id2vec = np.concatenate((embedding_matrix, np.random.randn(embedding_size-embedding_matrix.shape[0], embedding_len)))

    q = Embedding(embedding_size, embedding_len, weights=[id2vec], input_length=max_seq_len, trainable=True)(q_input)
    q = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode="sum")(q)
    q = Dropout(dropout)(q)
    q = Conv1D(filter_size, kernel_size, padding='same', activation='relu', strides=1)(q)
    q = MaxPooling1D(pooling_size)(q)
    q = Dropout(dropout)(q)
    q = Conv1D(filter_size, kernel_size, padding='same', activation='relu', strides=1)(q)
    q = MaxPooling1D(pooling_size)(q)
    q = Flatten()(q)
    q = Dropout(dropout)(q)
    q = BatchNormalization()(q)
    q = Dense(n_classes, activation='softmax')(q)
 
    model =  Model(inputs=q_input, outputs=q)
    return model

