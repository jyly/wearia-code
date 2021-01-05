# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.backend import *

def attentation_module(inputs,inshape):
    attention_probs = Dense(inshape, activation='softmax')(inputs)
    attention_mul =Multiply()([inputs, attention_probs])
    return attention_mul
    
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




def mlp_network_att(input_shape):
    lens=int(int(input_shape[0])*int(input_shape[1]))
    inputs = Input(shape=(input_shape), name='input')
    
    x = Flatten()(inputs)
    x = attentation_module(x,lens)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', name='output')(x)

    outputs=x
    return Model(inputs, outputs)

def lstm_network_1(input_shape):
    a=int(input_shape[0])
    b=int(input_shape[1])
    inputs = Input(shape=(input_shape), name='input')
    
    x = Reshape((a,b))(inputs)
    x = Permute((2,1))(x)
    x = CuDNNLSTM(units=32,input_shape=(b,a))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    
    outputs=x
    return Model(inputs,outputs)

def lstm_network_2(input_shape):
    a=int(input_shape[0])
    b=int(input_shape[1])
    inputs = Input(shape=(input_shape), name='input')
    
    x=Reshape((a,b))(inputs)
    x = Permute((2,1))(x)
    x = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,input_shape=(b,a))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)

def conv_network(input_shape):
    inputs = Input(shape=input_shape, name='input')

    x = Reshape((int(input_shape[0]),int(input_shape[1]),1))(inputs)
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 

    x = Conv2D(4, (1,3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(1, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)


def SBNN(input_shape):
    inputs = Input(shape=(input_shape), name='input')
    x = Conv2D(16, (1, 3),activation='relu',padding='same')(inputs)
    x = Conv2D(16, (1, 3),activation='relu',padding='same')(x)

    x = MaxPooling2D((1,3),strides=(1,2),padding='same')(x)

    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((1,2),strides=(1,2))(x)

    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((1,2),strides=(1,2))(x)
    
    x = Dense(1, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    outputs=x
    return Model(inputs, outputs)

def SCNN(input_shape):
    inputs = Input(shape=(input_shape), name='input')
    x1 = Conv2D(4, (1, 9),activation='relu',padding='same')(inputs)
    x = Conv2D(4, (1, 1),activation='relu',padding='same')(x1)

    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Concatenate()([x1,x])
    x = Conv2D(4, (1, 1),activation='relu',padding='same')(x)

    x = MaxPooling2D((1,3),strides=(1,2),padding='same')(x)

    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((1,3),strides=(1,2),padding='same')(x)

    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4, (1, 1),activation='relu',padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    outputs=x
    return Model(inputs, outputs)


def DCNN(input_shape):

    inputs = Input(shape=input_shape, name='input')
    x = Conv2D(16, (1, 3),activation='relu',padding='same')(inputs)
    x = Conv2D(16, (1, 3),activation='relu',padding='same')(x)

    x = MaxPooling2D((1,3),strides=(1,2),padding='same')(x)

    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((1,2),strides=(1,2))(x)

    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y1 = Conv2D(16, (1, 9),activation='relu',padding='same')(inputs)
    y = Conv2D(16, (1, 1),activation='relu',padding='same')(y1)

    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Concatenate()([y1,y])
    y = Conv2D(16, (1, 1),activation='relu',padding='same')(y)

    y = MaxPooling2D((1,3),strides=(1,2),padding='same')(y)

    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((1,3),strides=(1,2),padding='same')(y)

    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(16, (1, 3),padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    x = Concatenate()([x,y])
    x = GlobalAveragePooling2D()(x)

    # x = Dense(128, activation='relu',name='output')(x)
    outputs=x
    return Model(inputs, outputs)


def conv_lstm_network_1(input_shape):
    a=int(input_shape[0])
    b=int(input_shape[1])
    inputs = Input(shape=input_shape, name='input')
    
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(1, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 

    x = Reshape((a,50))(x)
    x = Permute((2,1))(x)

    x = CuDNNLSTM(units=32,input_shape=(50,a),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,input_shape=(50,a))(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)





def conv_lstm_network_2(input_shape):
    a=int(input_shape[0])
    b=int(input_shape[1])    
    inputs = Input(shape=input_shape, name='input')
    
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (a, 1), activation='relu')(x)
    x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(1, 4))(x) 
    newshapes=(50,4)
    x = Reshape(newshapes)(x)
    
    x = CuDNNLSTM(units=32,input_shape=(newshapes),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,input_shape=(newshapes))(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)

def slice(x,index):
    return x[:,index]

def conv_lstm_network_3(input_shape):
    inputs = Input(shape=input_shape, name='input')
    
    x = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(4, (1, 5), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 
    newshapes=(50,4)
    x = Reshape((int(input_shape[0]),50,4))(x)
    
    x1 = Lambda(slice,output_shape=(50,4),arguments={'index':0})(x)
    x2 = Lambda(slice,output_shape=(50,4),arguments={'index':1})(x)
    x3 = Lambda(slice,output_shape=(50,4),arguments={'index':2})(x)
    x4 = Lambda(slice,output_shape=(50,4),arguments={'index':3})(x)
    x5 = Lambda(slice,output_shape=(50,4),arguments={'index':4})(x)
    x6 = Lambda(slice,output_shape=(50,4),arguments={'index':5})(x)
    x7 = Lambda(slice,output_shape=(50,4),arguments={'index':6})(x)
    x8 = Lambda(slice,output_shape=(50,4),arguments={'index':7})(x)

    x1 = CuDNNLSTM(units=32,input_shape=(newshapes))(x1)
    x2 = CuDNNLSTM(units=32,input_shape=(newshapes))(x2)
    x3 = CuDNNLSTM(units=32,input_shape=(newshapes))(x3)
    x4 = CuDNNLSTM(units=32,input_shape=(newshapes))(x4)
    x5 = CuDNNLSTM(units=32,input_shape=(newshapes))(x5)
    x6 = CuDNNLSTM(units=32,input_shape=(newshapes))(x6)
    x7 = CuDNNLSTM(units=32,input_shape=(newshapes))(x7)
    x8 = CuDNNLSTM(units=32,input_shape=(newshapes))(x8)
    
    # x = Concatenate()([x1,x2])
    x = Concatenate()([x1,x2,x3,x4,x5,x6,x7,x8])
    # x = Concatenate()([x1,x2,x3,x4,x5,x6])

    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)

def conv_pic_network(input_shape):
    inputs = Input(shape=input_shape, name='input')
    
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(4, 4))(x) 
    x = Conv2D(32, (3,3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(8, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)



def single_conv_lstm(x):
    x = Conv1D(8,3, activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(8, 3, activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(1, 1, activation='relu',padding='same')(x)


    y1=CuDNNLSTM(units=1,input_shape=(200,1),return_sequences=True)(x)
    y1=CuDNNLSTM(units=1,input_shape=(200,1),return_sequences=True)(y1)

    att = Add()([x,y1])
    att = Dense(1, activation='tanh')(att)
    att = BatchNormalization()(att)
    # att = Reshape((200,1))(att)

    y = Multiply()([y1,att])
    return y
    # return Model(x,y)




def conv_lstm_4(input_shape):
    inputs = Input(shape=input_shape, name='input')
    x = inputs

    x1 = Lambda(slice,output_shape=(200,1),arguments={'index':0})(x)
    x2 = Lambda(slice,output_shape=(200,1),arguments={'index':1})(x)
    x3 = Lambda(slice,output_shape=(200,1),arguments={'index':2})(x)
    x4 = Lambda(slice,output_shape=(200,1),arguments={'index':3})(x)
    x5 = Lambda(slice,output_shape=(200,1),arguments={'index':4})(x)
    x6 = Lambda(slice,output_shape=(200,1),arguments={'index':5})(x)

    x1 = single_conv_lstm(x1)
    x2 = single_conv_lstm(x2)
    x3 = single_conv_lstm(x3)
    x4 = single_conv_lstm(x4)
    x5 = single_conv_lstm(x5)
    x6 = single_conv_lstm(x6)

    x = Concatenate()([x1,x2,x3,x4,x5,x6])
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)

    outputs=x
    return Model(inputs,outputs)


def single_pyramid(x):
    x=Conv1D(8,3,activation='relu',padding='same')(x)
    x1 = Dropout(0.1)(x)
    x1 = MaxPooling1D(pool_size=2)(x1) 
    x2 = Conv1D(8, 5, activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x3 = Conv1D(8, 3, activation='relu',padding='same')(x2)
    x3 = Dropout(0.1)(x3)
    x3 = MaxPooling1D(pool_size=2)(x3)

    y3 = Conv1D(1,1, activation='relu',padding='same')(x3)

    temp = UpSampling1D(size=2)(y3)

    y2 = Conv1D(1, 1, activation='relu',padding='same')(x2)
    y2=Add()([y2,temp])
   
    temp = UpSampling1D(size=2)(y2)
    y1 = Conv1D(1, 1, activation='relu',padding='same')(x1)
    y1 =Add()([y1,temp])

    y1 = CuDNNLSTM(units=1,input_shape=(100,1),return_sequences=True)(y1)
    y1 = CuDNNLSTM(units=1,input_shape=(100,1),return_sequences=True)(y1)
    y = MaxPooling1D(pool_size=4)(y1)

    return y




# def pyramid_1(input_shape):
  
#     inputs = Input(shape=input_shape, name='input')
    
#     x=inputs
#     x1 = Lambda(slice,output_shape=(200,1),arguments={'index':0})(x)
#     x2 = Lambda(slice,output_shape=(200,1),arguments={'index':1})(x)
#     # x3 = Lambda(slice,output_shape=(200,1),arguments={'index':2})(x)
#     # x4 = Lambda(slice,output_shape=(200,1),arguments={'index':3})(x)
#     # x5 = Lambda(slice,output_shape=(200,1),arguments={'index':4})(x)
#     # x6 = Lambda(slice,output_shape=(200,1),arguments={'index':5})(x)

#     x1 = single_pyramid(x1)
#     x2 = single_pyramid(x2)
#     # x3 = single_pyramid(x3)
#     # x4 = single_pyramid(x4)
#     # x5 = single_pyramid(x5)
#     # x6 = single_pyramid(x6)

#     # x = Concatenate()([x1,x2,x3,x4,x5,x6])
#     x = Concatenate()([x1,x2])
   
#     x = Flatten()(x)
#     x = Dense(128, activation='relu',name='output')(x)
    
#     outputs=x

#     return Model(inputs, outputs)


def convlstm2D(input_shape):
    a=int(input_shape[0])
    b=int(int(input_shape[1]))
    inputs = Input(shape=input_shape, name='input')
    y1=Reshape((a,b,1,1))(inputs)

    y1 = ConvLSTM2D(filters=32,kernel_size=3,padding='same',return_sequences=True)(y1)
    y1 = BatchNormalization()(y1)
    y1 = ConvLSTM2D(filters=32,kernel_size=3,padding='same')(y1)
    y1 = BatchNormalization()(y1)

    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)



def pyramid_1(input_shape):
    a=int(input_shape[0])
    b=int(int(input_shape[1]/2))
    inputs = Input(shape=input_shape, name='input')
    
    x1 = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(4, (1, 3), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(4, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.2)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    y3 = Conv2D(4, (1, 1), activation='relu',padding='same')(x3)
    
    temp = UpSampling2D(size=(1,2))(x3)
    y2 = Conv2D(4, (1, 1), activation='relu',padding='same')(x2)
    y2=Concatenate()([x2,temp])
    y2=Conv2D(4, (1, 1), activation='relu',padding='same')(y2)

    temp = UpSampling2D(size=(1,2))(y2)
    y1 = Conv2D(4, (1, 1), activation='relu',padding='same')(x1)
    y1=Concatenate()([x1,temp])
    y1=Conv2D(4, (1, 1), activation='relu',padding='same')(y1)


    y1=Permute((2,1,3))(y1)
    y1=Reshape((b,int(a*4)))(y1)

    y1 = CuDNNGRU(units=32,return_sequences=True)(y1)
    y1 = Dropout(0.3)(y1)
    y1 = CuDNNGRU(units=32)(y1)
    
    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)


def pyramid_2(input_shape):
    a=int(input_shape[0])
    b=int(int(input_shape[1]/2))
    inputs = Input(shape=input_shape, name='input')
    
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(inputs)
    x1 = Dropout(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

    x2 = Conv2D(4, (1, 3), activation='relu',padding='same')(x1)
    x2 = Dropout(0.1)(x2)
    x2 = MaxPooling2D(pool_size=(1, 2))(x2)

    x3 = Conv2D(4, (1, 3), activation='relu',padding='same')(x2)
    x3 = Dropout(0.2)(x3)
    x3 = MaxPooling2D(pool_size=(1, 2))(x3)

    # y3 = Conv2D(2, (1, 1), activation='relu',padding='same')(x3)
    
    temp = UpSampling2D(size=(1,2))(x3)
    # y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
    # y2=Add()([y2,temp])
    y2=Concatenate()([x2,temp])
    y2=Conv2D(8, (1, 1), activation='relu',padding='same')(y2)

    temp = UpSampling2D(size=(1,2))(y2)
    # y1 = Conv2D(8, (1, 1), activation='relu',padding='same')(x1)
    # y1 =Add()([y1,temp])
    y1=Concatenate()([x1,temp])
    y1=Conv2D(16, (1, 1), activation='relu',padding='same')(y1)

    # y1 = Conv2D(4, (1, 3), activation='relu',padding='same')(y1)
    # y1 = Conv2D(4, (1, 3), activation='relu',padding='same')(y1)

    # y1=Reshape((a,b))(y1)
    # y1=Permute((2,1))(y1)
    y1=Permute((2,1,3))(y1)
    y1=Reshape((b,int(a*16)))(y1)

    y1 = CuDNNGRU(units=32,return_sequences=True)(y1)
    y1 = Dropout(0.3)(y1)
    y1 = CuDNNGRU(units=32)(y1)
    
    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)

# def pyramid_1(input_shape):
#     a=int(input_shape[0])
#     b=int(int(input_shape[1]/2))
#     inputs = Input(shape=input_shape, name='input')
    
#     x1 = Conv2D(4, (1, 3), activation='relu',padding='same')(inputs)

#     x1 = Conv2D(4, (1, 3), activation='relu',padding='same')(x1)
#     x1 = MaxPooling2D(pool_size=(1, 2))(x1) 
  

#     x2 = Conv2D(4, (1, 3), activation='relu',padding='same')(x1)
#     x2 = Conv2D(4, (1, 3),padding='same')(x2)
#     x1 = Dropout(0.1)(x1)
#     x2 = MaxPooling2D(pool_size=(1, 2))(x2)

#     x3 = Conv2D(2, (1, 3), activation='relu',padding='same')(x2)
#     x3 = Conv2D(2, (1, 3), activation='relu',padding='same')(x3)
#     x1 = Dropout(0.2)(x1)
#     x3 = MaxPooling2D(pool_size=(1, 2))(x3)

#     y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
    
#     temp = UpSampling2D(size=(1,2))(y3)
#     y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
#     y2=Add()([y2,temp])
   
#     temp = UpSampling2D(size=(1,2))(y2)
#     y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
#     y1 =Add()([y1,temp])


#     # y1 = Conv2D(4, (1, 3), activation='relu',padding='same')(y1)
#     # y1 = Conv2D(4, (1, 3), activation='relu',padding='same')(y1)



#     y1=Reshape((a,b))(y1)
#     # y1=Permute((2,1,3))(y1)

#     # y1 = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(y1)
#     # y1 = CuDNNLSTM(units=32,input_shape=(b,a))(y1)

#     y1 = CuDNNGRU(units=32,return_sequences=True)(y1)
#     y1 = Dropout(0.3)(y1)
#     y1 = CuDNNGRU(units=32)(y1)
    

#     # y1=Reshape((1,a,b,1))(y1)
#     # print(y1)
#     # y1 = ConvLSTM2D(filters=6,kernel_size=(1,3),padding='same',return_sequences=True)(y1)
#     # y1 = BatchNormalization()(y1)
#     # y1 = ConvLSTM2D(filters=6,kernel_size=(1,3),padding='same',)(y1)
#     # y1 = BatchNormalization()(y1)



#     x = Flatten()(y1)
#     x = Dense(128, activation='relu',name='output')(x)
    
#     outputs=x
#     return Model(inputs, outputs)




# def pyramid_2(input_shape):
#     inputs = Input(shape=input_shape, name='input')
    
#     x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(inputs)
#     x1 = Dropout(0.1)(x1)
#     x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

#     x2 = Conv2D(8, (1, 5), activation='relu',padding='same')(x1)
#     x2 = Dropout(0.1)(x2)
#     x2 = MaxPooling2D(pool_size=(1, 2))(x2)

#     x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
#     x3 = Dropout(0.1)(x3)
#     x3 = MaxPooling2D(pool_size=(1, 2))(x3)

#     y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)

#     temp = UpSampling2D(size=(1,2))(y3)
#     y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
#     y2=Add()([y2,temp])
    
#     temp = UpSampling2D(size=(1,2))(y2)
#     y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
#     y1 =Add()([y1,temp])

#     y1=Reshape((2,100))(y1)
#     y1=Permute((2,1))(y1)

#     y2=Reshape((2,50))(y2)
#     y2=Permute((2,1))(y2)

#     y3=Reshape((2,25))(y3)
#     y3=Permute((2,1))(y3)

#     y1 = CuDNNLSTM(units=32,input_shape=(100,2),return_sequences=True)(y1)
#     y1 = CuDNNLSTM(units=32,input_shape=(100,2))(y1)
    
#     y2 = CuDNNLSTM(units=32,input_shape=(50,2),return_sequences=True)(y2)    
#     y2 = CuDNNLSTM(units=32,input_shape=(50,2))(y2)
    
#     y3 = CuDNNLSTM(units=32,input_shape=(25,2),return_sequences=True)(y3)
#     y3 = CuDNNLSTM(units=32,input_shape=(25,2))(y3)

#     x =concatenate([y1,y2,y3])
#     x = Dense(128, activation='relu',name='output')(x)
    
#     outputs=x
#     return Model(inputs, outputs)


# def pyramid_3(input_shape):
#     inputs = Input(shape=input_shape, name='input')
    
#     x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(inputs)
#     x1 = Dropout(0.1)(x1)
#     x1 = MaxPooling2D(pool_size=(1, 2))(x1) 

#     x2 = Conv2D(8, (1, 5), activation='relu',padding='same')(x1)
#     x2 = Dropout(0.1)(x2)
#     x2 = MaxPooling2D(pool_size=(1, 2))(x2)

#     x3 = Conv2D(8, (1, 3), activation='relu',padding='same')(x2)
#     x3 = Dropout(0.1)(x3)
#     x3 = MaxPooling2D(pool_size=(1, 2))(x3)

#     y3 = Conv2D(1, (1, 1), activation='relu',padding='same')(x3)
#     y3=Reshape((2,25))(y3)
#     y3=Permute((2,1))(y3)
#     y3 = CuDNNLSTM(units=2,input_shape=(25,2),return_sequences=True)(y3)
#     y3=Permute((2,1))(y3)
#     y3=Reshape((2,25,1))(y3)

#     temp = UpSampling2D(size=(1,2))(y3)
#     y2 = Conv2D(1, (1, 1), activation='relu',padding='same')(x2)
#     y2=Add()([y2,temp])
#     y2=Reshape((2,50))(y2)
#     y2=Permute((2,1))(y2)
#     y2 = CuDNNLSTM(units=2,input_shape=(50,2),return_sequences=True)(y2)
#     y2=Permute((2,1))(y2)
#     y2=Reshape((2,50,1))(y2)


#     temp = UpSampling2D(size=(1,2))(y2)
#     y1 = Conv2D(1, (1, 1), activation='relu',padding='same')(x1)
#     y1 =Add()([y1,temp])
#     y1=Reshape((2,100))(y1)
#     y1=Permute((2,1))(y1)
# #     y1 = CuDNNLSTM(units=32,input_shape=(100,2))(y1)
#     y1 = CuDNNLSTM(units=2,input_shape=(100,2),return_sequences=True)(y1)
#     x = Flatten()(y1)
# #     x=y1
#     x = Dense(128, activation='relu',name='output')(x)
    
#     outputs=x
#     return Model(inputs, outputs)


#batch 4086 迭代20次
def pyramid_ppg(input_shape):
    a=int(input_shape[0])
    b=int(int(input_shape[1]/2))
    inputs = Input(shape=input_shape, name='input')
    
    x1 = Conv2D(8, (1, 3), activation='relu',padding='same')(inputs)
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

    y1=Reshape((a,b))(y1)
    y1=Permute((2,1))(y1)
    y1 = CuDNNLSTM(units=32,input_shape=(b,a),return_sequences=True)(y1)
    y1 = CuDNNLSTM(units=32,input_shape=(b,a))(y1)
    x = Flatten()(y1)
    x = Dense(128, activation='relu',name='output')(x)
    # x = Dense(32, activation='relu',name='output')(x)
    
    outputs=x
    return Model(inputs, outputs)
