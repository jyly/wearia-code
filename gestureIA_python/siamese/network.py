# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def attentation_module(inputs,inshape):
    attention_probs = Dense(inshape, activation='softmax')(inputs)
    attention_mul =Multiply()([inputs, attention_probs])
    return attention_mul

    
def mlp_network(input_shape):
    input = Input(shape=(input_shape), name='input')
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    return Model(input, x)



def mlp_network_att(input_shape):
    input = Input(shape=(input_shape), name='input')
    x = Flatten()(input)

    x=attentation_module(x,80)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    # x=attentation_module(x,128)

    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x=attentation_module(x,128)

    x = Dense(128, activation='relu', name='output')(x)
    # x = Dense(2, activation='relu', name='output')(x)
    # x = Dense(2, activation='softmax', name='output')(x)
    # x = Dense(128, name='output')(x)

    return Model(input, x)





def lstm_network(input_shape):
    input = Input(shape=(input_shape), name='input')
    x=LSTM(units=30,input_shape=(input_shape))(input)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    return Model(input, x)


def conv_network(input_shape):
    input = Input(shape=input_shape, name='input')
    # x = Conv2D(4, (1, 5), activation='relu',padding='same')(input)
    # x = Dropout(0.1)(x)
    # x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)

    x = Conv2D(4, (1, 3), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
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



def conv_lstm_network_1(input_shape):
    inputs = Input(shape=input_shape, name='input')
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(1, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)


    x = MaxPooling2D(pool_size=(1, 4))(x) 
    # x = Conv2D(4, (1,5), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    newreshapes=(int(input_shape[0]),50)
    newshapes=(50,int(input_shape[0]))
    print(newreshapes)
    print(newshapes)
    x = Reshape(newreshapes)(x)
    x = Permute((2,1))(x)

    x = LSTM(units=32,input_shape=(newshapes),return_sequences=True)(x)
    x = LSTM(units=32,input_shape=(newshapes))(x)

    # x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(inputs, x)


def conv_lstm_network_2(input_shape):
    input = Input(shape=input_shape, name='input')
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(4, (int(input_shape[0]), 1), activation='relu')(x)
    x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(1, 4))(x) 
    # x = Conv2D(4, (1,5), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    newshapes=(50,4)
    print(newshapes)
    x = Reshape(newshapes)(x)

    x = LSTM(units=32,input_shape=(newshapes),return_sequences=True)(x)
    x = LSTM(units=32,input_shape=(newshapes))(x)

    # x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)

def slice(x,index):
    return x[:,index]

def conv_lstm_network_3(input_shape):
    input = Input(shape=input_shape, name='input')
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 
    # x = Conv2D(4, (1,5), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    newshapes=(50,4)
    x = Reshape((int(input_shape[0]),50,4))(x)
    # print(x)
    # print(x[:,0])
    x1 = Lambda(slice,output_shape=(50,4),arguments={'index':0})(x)
    x2 = Lambda(slice,output_shape=(50,4),arguments={'index':1})(x)

    x1 = LSTM(units=32,input_shape=(newshapes))(x1)
    x2 = LSTM(units=32,input_shape=(newshapes))(x2)
    # print(x1)
    x = Concatenate()([x1,x2])

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

