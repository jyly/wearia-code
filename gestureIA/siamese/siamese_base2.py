# -*- coding=utf-8 -*-
import os     
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["PATH"] += os.pathsep + 'E:/system/python/graphviz/bin'
import tensorflow as tf
import keras
import numpy as np
import random
from keras.callbacks import TensorBoard
# from keras.datasets import manhattan_distance
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D
from keras import backend as K
from keras import regularizers,optimizers
from keras.models import load_model,model_from_json
def mlp_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=(input_shape))
    # x = Flatten()(input)
    x = input
    #全连接层
    x = Dense(256, activation='relu')(x)
    #遗忘层
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)


    return Model(input, x)


def triplet_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    # print('y_pred.shape = ', y_pred)
    # print('y_true.shape = ', y_true)
    
    alpha=0.4
    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    # pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=1))
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    # neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=1))
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, K.epsilon())

    return loss

def create_siamese_network(input_shape):
    base_network = mlp_network(input_shape)
    input_a = Input(shape=input_shape, name='input1')
    input_b = Input(shape=input_shape, name='input2')
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    merged_soft = concatenate([processed_a, processed_b], axis=-1, name='output')
    model = Model(inputs=[input_a, input_b], outputs=merged_soft)
    model.summary()
    rms = optimizers.RMSprop()
    model.compile(loss=["categorical_crossentropy", siamese_loss],optimizer=rms, metrics=["accuracy"])
    return model