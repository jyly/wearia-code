# -*- coding=utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation,ZeroPadding2D,BatchNormalization,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D,Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D,Reshape,BatchNormalization,Activation,Add,Multiply,Dot,Average,Concatenate,LSTM,Reshape,Permute,Lambda,RepeatVector
from keras import optimizers
import keras.layers as KL
import keras.backend as K
from keras.layers.core import Layer
from keras.initializers import glorot_uniform


    
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

def conv_pic_network(input_shape):

    input = Input(shape=input_shape, name='input')

    x = Conv2D(32, (3, 3), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(4, 4))(x) 
    # x = Conv2D(4, (1, 10), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)
   
    x = Conv2D(32, (3,3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)
    # x = MaxPooling2D(pool_size=(1, 5))(x)
    # x = Conv2D(32, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(8, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)


def identity_block(X, f, filters, stage, block):
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    F1, F2, F3 = filters
 
    X_shortcut = X
 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
 
    return X


def convolution_block(X, f, filters, stage, block, s=2):
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
 
    X_shortcut = X
 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
 
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut)
 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
 
    return X



def ResNet50(input_shape = (32, 32, 2)):
 
    X_input = Input(input_shape)
 
    # X = ZeroPadding2D((3, 3))(X_input)
 
    X = Conv2D(64, (7, 7), strides = (2,2), padding='same',name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis= 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    # X = MaxPooling2D((3, 3), strides = (2,2))(X)
 
    X = convolution_block(X, f = 3, filters = [64,64,256], stage = 2, block = 'a', s = 1)
    X = identity_block(X, 3, [64,64,256], stage=2, block='b')
    X = identity_block(X, 3, [64,64,256], stage=2, block='c')
 
    X = convolution_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')
 
    X = convolution_block(X, f = 3, filters = [256,256,1024], stage = 4, block = 'a', s = 2)
    X = identity_block(X, 3, [256,256,1024], stage=4, block='b')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='c')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='d')    
    X = identity_block(X, 3, [256,256,1024], stage=4, block='e')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='f')
 
    X = convolution_block(X, f = 3, filters = [512,512,2048], stage = 5, block = 'a', s = 2)
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')
 
    # X = AveragePooling2D((2, 2), name='avg_pool')(X)
 
    X = Flatten()(X)
    X = Dense(128, activation = 'softmax', name = 'fc' , kernel_initializer = glorot_uniform(seed=0))(X)
 
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
 
    return model