# -*- coding=utf-8 -*-
import keras
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D,Reshape,BatchNormalization,Activation
from keras import backend as K
from keras import regularizers,optimizers

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.maximum(K.sqrt(sum_square), K.epsilon())

def manhattan_distance(vects):
    x, y = vects
    sum_square = K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))
    return K.maximum(sum_square, K.epsilon())

def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    sum_square=K.sum(x * y, axis=-1, keepdims=True)
    return K.maximum(sum_square, K.epsilon())


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss_1(y_true, y_pred):#实际值，预测值,全部正确时返回1，全部错误时返回0
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def contrastive_loss_2(y_true, y_pred):#实际值，预测值,全部正确时返回1，全部错误时返回0
    Q=5
    sqaure_pred = K.square(y_pred)
    exp_pred=K.exp(-(float(2.77/Q)*y_pred))
    return K.mean((y_true)*float(2/Q)*sqaure_pred+(1-y_true)*2*Q*exp_pred)


def accuracy(y_true, y_pred): # Tensor上的操作

    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)



def mlp_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
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
    # x = Dense(128, name='output')(x)

    return Model(input, x)



def conv_network(input_shape):

    input = Input(shape=input_shape, name='input')
    # x = Reshape((2,300,1))(input)


    # x = Conv2D(4, (2, 1))(input)
    # x = BatchNormalization(epsilon=1e-06)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.1)(x)

    x = Conv2D(4, (1, 3), activation='relu')(input)
    x = Dropout(0.1)(x)


    x = Conv2D(8, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(1, 4))(x) 
    # x = Conv2D(4, (1, 10), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)
   
    x = Conv2D(8, (1,5), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(1, 5))(x)
    # x = Conv2D(32, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(2, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu',name='output')(x)
    return Model(input, x)





def create_siamese_network(input_shape):
    
    base_network = mlp_network(input_shape)
    # base_network = conv_network(input_shape)
    base_network.summary()
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    model.summary()

    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    return model,base_network


# def create_presudo_siamese_network(input_shape):
    
#     # base_network = conv_network(input_shape)
#     # base_network_1 = mlp_network(input_shape)

#     input_a = Input(shape=input_shape)
#     input_b = Input(shape=input_shape)
#     # because we re-use the same instance `base_network`,
#     # the weights of the network
#     # will be shared across the two branches
#     processed_a = base_network_1(input_a)
#     processed_b = base_network_2(input_b)

#     distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

#     model = Model([input_a, input_b], distance)
#     # keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
#     model.summary()

#     # train
#     rms = optimizers.RMSprop()
#     model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
#     return model


# def cwt_network(input_shape):
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     input = Input(shape=input_shape)
 
#     # x = MaxPooling2D(pool_size=(2, 2))(input)
#     x = Conv2D(16, (2, 2), activation='relu',padding='same')(input)
#     x = Dropout(0.1)(x)

#     # x = MaxPooling2D(pool_size=(2, 2))(input)
#     x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
#     x = Dropout(0.1)(x)

#     # x = MaxPooling2D(pool_size=(2, 2))(input)
#     x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
#     x = Dropout(0.2)(x)

#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)

#     return Model(input, x)

