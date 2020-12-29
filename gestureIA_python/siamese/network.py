# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.backend import *

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





def lstm_network_1(input_shape):
    input = Input(shape=(input_shape), name='input')
    newshapes=(int(input_shape[0]),int(input_shape[1]))
    x=Reshape(newshapes)(input)
    newreshapes=(int(input_shape[1]),int(input_shape[0]))
    x = Permute((2,1))(x)
    x=LSTM(units=32,input_shape=(newreshapes))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    return Model(input, x)


def lstm_network_2(input_shape):
    input = Input(shape=(input_shape), name='input')
    newshapes=(int(input_shape[0]),int(input_shape[1]))
    x=Reshape(newshapes)(input)
    newreshapes=(int(input_shape[1]),int(input_shape[0]))
    x = Permute((2,1))(x)
    x = LSTM(units=32,input_shape=(newshapes),return_sequences=True)(x)
    x = LSTM(units=32,input_shape=(newshapes),return_sequences=True)(x)
    x = LSTM(units=32,input_shape=(newshapes))(x)
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
    x3 = Lambda(slice,output_shape=(50,4),arguments={'index':2})(x)
    x4 = Lambda(slice,output_shape=(50,4),arguments={'index':3})(x)
    x5 = Lambda(slice,output_shape=(50,4),arguments={'index':4})(x)
    x6 = Lambda(slice,output_shape=(50,4),arguments={'index':5})(x)
    x7 = Lambda(slice,output_shape=(50,4),arguments={'index':6})(x)
    x8 = Lambda(slice,output_shape=(50,4),arguments={'index':7})(x)

    x1 = LSTM(units=32,input_shape=(newshapes))(x1)
    x2 = LSTM(units=32,input_shape=(newshapes))(x2)
    x3 = LSTM(units=32,input_shape=(newshapes))(x3)
    x4 = LSTM(units=32,input_shape=(newshapes))(x4)
    x5 = LSTM(units=32,input_shape=(newshapes))(x5)
    x6 = LSTM(units=32,input_shape=(newshapes))(x6)
    x7 = LSTM(units=32,input_shape=(newshapes))(x7)
    x8 = LSTM(units=32,input_shape=(newshapes))(x8)
    # print(x1)
    # x = Concatenate()([x1,x2])
    x = Concatenate()([x1,x2,x3,x4,x5,x6,x7,x8])
    # x = Concatenate()([x1,x2,x3,x4,x5,x6])

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





def pyramid_1(input_shape):
    input = Input(shape=input_shape, name='input')
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(input)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(8, (1, 5), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    y3 = UpSampling2D(size=(1,2))(y3)
    print(y3)

    y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    y2=Add()([y2,y3])
    y2 = UpSampling2D(size=(1,2))(y2)

    y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
    y1 =Add()([y1,y2])

    a=int(input_shape[0])
    b=int(int(input_shape[1]/2))

    y1=Reshape((a,b))(y1)
    y1=Permute((2,1))(y1)
    print(y1)
    y1 = LSTM(units=a,input_shape=(b,a),return_sequences=True)(y1)
    print(y1)
    y1 = LSTM(units=32,input_shape=(b,a))(y1)
    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)



def pyramid_2(input_shape):
    input = Input(shape=input_shape, name='input')
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(input)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(8, (1, 5), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    temp = UpSampling2D(size=(1,2))(y3)
    print(y3)

    y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    y2=Add()([y2,temp])
    temp = UpSampling2D(size=(1,2))(y2)

    y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
    y1 =Add()([y1,temp])


    y1=Reshape((2,100))(y1)
    y1=Permute((2,1))(y1)


    y2=Reshape((2,50))(y2)
    y2=Permute((2,1))(y2)

    y3=Reshape((2,25))(y3)
    y3=Permute((2,1))(y3)


    y1 = LSTM(units=32,input_shape=(100,2))(y1)
    y2 = LSTM(units=32,input_shape=(50,2))(y2)
    y3 = LSTM(units=32,input_shape=(25,2))(y3)

    x =concatenate([y1,y2,y3])

    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)



def pyramid_3(input_shape):
    input = Input(shape=input_shape, name='input')
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(input)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(8, (1, 5), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    print(y3)
    y3=Reshape((2,25))(y3)
    y3=Permute((2,1))(y3)
    y3 = LSTM(units=2,input_shape=(25,2),return_sequences=True)(y3)
    y3=Permute((2,1))(y3)
    y3=Reshape((2,25,1))(y3)



    temp = UpSampling2D(size=(1,2))(y3)
    y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    y2=Add()([y2,temp])
    y2=Reshape((2,50))(y2)
    y2=Permute((2,1))(y2)
    y2 = LSTM(units=2,input_shape=(50,2),return_sequences=True)(y2)
    y2=Permute((2,1))(y2)
    y2=Reshape((2,50,1))(y2)


    temp = UpSampling2D(size=(1,2))(y2)
    y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
    y1 =Add()([y1,temp])
    y1=Reshape((2,100))(y1)
    y1=Permute((2,1))(y1)
    # y1 = LSTM(units=32,input_shape=(100,2))(y1)
    # x=y1
  

    y1 = LSTM(units=32,input_shape=(100,2),return_sequences=True)(y1)
    x = Flatten()(y1)

    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)