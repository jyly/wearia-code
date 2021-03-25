# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.models import *
from tensorflow.keras.backend import *

    
def bilstm_network(input_shape):
    inputs = Input(shape=(input_shape), name='input')

    x = Permute((2,1))(inputs)
    x = Bidirectional(CuDNNLSTM(8, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='output')(x)

    outputs=x
    return Model(inputs, outputs)

def lstm_network(input_shape):
    inputs = Input(shape=(input_shape), name='input')
    
    x = Permute((2,1))(inputs)
    x = CuDNNLSTM(units=32,return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = CuDNNLSTM(units=32)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='output')(x)
    
    outputs=x
    return Model(inputs,outputs)


def mlp_network(input_shape):
    inputs = Input(shape=(input_shape), name='input')
    
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)




def conv_network(input_shape):
    inputs = Input(shape=input_shape, name='input')

    x = Reshape((int(input_shape[0]),int(input_shape[1]),1))(inputs)
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    # x = BatchNormalization(axis=2)(x)
    # x = Activation('relu')(x) 
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x) 
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x) 
    x = Conv2D(4, (1,3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(1, (1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)


def conv_lstm_network(input_shape):

    inputs = Input(shape=input_shape, name='input')

    x = Reshape((int(input_shape[0]),int(input_shape[1]),1))(inputs)
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x) 
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x) 
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(1, (1, 1), activation='relu',padding='same')(x)

    a=int(input_shape[0])
    b=int(int(input_shape[1])/4)
    x = Reshape((a,b))(x)
    x = Permute((2,1))(x)

    x = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,input_shape=(b,a))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)




def pyramid_network(input_shape):
    inputs = Input(shape=input_shape, name='input')

    x = Reshape((int(input_shape[0]),int(input_shape[1]),1))(inputs)
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(x)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(8, (1, 3), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    
    temp = UpSampling2D(size=(1,2))(y3)
    y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    y2=Add()([y2,temp])
   
    temp = UpSampling2D(size=(1,2))(y2)
    y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
    y1 =Add()([y1,temp])

    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)


#最高精度方案
def pyramid_lstm_network(input_shape):
    inputs = Input(shape=input_shape, name='input')
    
    x = Reshape((int(input_shape[0]),int(input_shape[1]),1))(inputs)
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(x)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(8, (1, 3), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    
    temp = UpSampling2D(size=(1,2))(y3)
    y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    y2=Add()([y2,temp])
   
    temp = UpSampling2D(size=(1,2))(y2)
    y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
    y1 =Add()([y1,temp])

    a=int(input_shape[0])
    b=int(int(input_shape[1]/2))
    y1=Reshape((a,b))(y1)
    y1=Permute((2,1))(y1)
    y1 = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(y1)
    y1 = CuDNNLSTM(units=32,input_shape=(b,a))(y1)
    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    outputs=x
    return Model(inputs, outputs)
