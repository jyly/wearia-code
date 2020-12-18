# -*- coding=utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D,Reshape,BatchNormalization,Activation,Add,Multiply,Dot,Average,Concatenate,LSTM,Reshape,Permute,Lambda,RepeatVector
from keras import optimizers
import keras.layers as KL
import keras.backend as K
from keras.layers.core import Layer


    
def mlp_network(input_shape):

    input = Input(shape=(input_shape), name='input')
    x = Flatten()(input)
    # x = BatchNormalization(epsilon=1e-06)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    # x = Dense(2, activation='relu', name='output')(x)
    # x = Dense(2, activation='softmax', name='output')(x)
    # x = Dense(128, name='output')(x)

    return Model(input, x)


def mlp_network_incre(input_shape):

    input = Input(shape=(input_shape), name='input')
    x = Flatten()(input)
    # x = BatchNormalization(epsilon=1e-06)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    # x = Dense(2, activation='relu', name='output')(x)
    # x = Dense(2, activation='softmax', name='output')(x)
    # x = Dense(128, name='output')(x)

    return Model(input, x)



def lstm_network(input_shape):
    input = Input(shape=(input_shape), name='input')
    # x = Flatten()(input)
    # x = BatchNormalization(epsilon=1e-06)(x)
    x=LSTM(units=30,input_shape=(input_shape))(input)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    # x = Dense(2, activation='relu', name='output')(x)
    # x = Dense(128, name='output')(x)

    return Model(input, x)


def conv_network(input_shape):

    input = Input(shape=input_shape, name='input')

    x = Conv2D(4, (1, 5), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)

    x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 
    # x = Conv2D(4, (1, 10), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)
   
    x = Conv2D(4, (1,5), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)
    # x = MaxPooling2D(pool_size=(1, 5))(x)
    # x = Conv2D(32, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(2, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)

