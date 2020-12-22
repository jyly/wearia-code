# -*- coding=utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D,Reshape,BatchNormalization,Activation,Add,Multiply,Dot,Average,Concatenate,LSTM,Reshape,Permute,Lambda,RepeatVector
from keras import optimizers
from siamese.siamese_tools import *
import keras.layers as KL
import keras.backend as K
from keras.layers.core import Layer
from siamese.network import *



def create_siamese_network(input_shape):
    
    # base_network = mlp_network(input_shape)
    # base_network = mlp_network_incre(input_shape)
    
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



def create_siamese_network_lstm(input_shape):
    
    # base_network = lstm_network(input_shape)
    base_network = lstm_network_att(input_shape)

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

def create_siamese_network_mlp(input_shape):
    
    base_network = mlp_network(input_shape)
    # base_network = mlp_network_att(input_shape)

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

def create_siamese_network_conv(input_shape):
    
    base_network = conv_network(input_shape)
    # base_network = ResNet50(input_shape)
    # base_network = conv_pic_network(input_shape)
    # base_network = creatresnet(input_shape)

    # base_network = conv_network_att(input_shape)
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

def create_mul_feature_siamese_network(input_shape_1,input_shape_2):
    
    base_network_1 = mlp_network(input_shape_1)
    base_network_2 = mlp_network(input_shape_2)

    input_a = Input(shape=input_shape_1)
    input_b = Input(shape=input_shape_1)
    processed_a = base_network_1(input_a)
    processed_b = base_network_1(input_b)
    
    input_c = Input(shape=input_shape_2)
    input_d = Input(shape=input_shape_2)
    processed_c = base_network_2(input_c)
    processed_d = base_network_2(input_d)

    concatenated_1 = Concatenate()([processed_a, processed_c])
    concatenated_2 = Concatenate()([processed_b, processed_d])

    x_1 = Dense(128, activation='relu')(concatenated_1)
    x_2 = Dense(128, activation='relu')(concatenated_2)

    # averages=Concatenate()([x_1, x_2])
    # distance = Dense(2, activation='softmax')(averages)
    # distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([x_1, x_2])
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([concatenated_1, concatenated_2])

    model = Model([input_a, input_b, input_c, input_d], distance)
    model.summary()
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model,base_network_1,base_network_2


def create_mul_data_siamese_network(input_shape_1,input_shape_2):
    
    base_network_1 = conv_network(input_shape_1)
    base_network_2 = conv_network(input_shape_2)

    input_a = Input(shape=input_shape_1)
    input_b = Input(shape=input_shape_1)
    processed_a = base_network_1(input_a)
    processed_b = base_network_1(input_b)
    
    input_c = Input(shape=input_shape_2)
    input_d = Input(shape=input_shape_2)
    processed_c = base_network_2(input_c)
    processed_d = base_network_2(input_d)

    concatenated_1 = Concatenate()([processed_a, processed_c])
    concatenated_2 = Concatenate()([processed_b, processed_d])

    x_1 = Dense(128, activation='relu')(concatenated_1)
    x_2 = Dense(128, activation='relu')(concatenated_2)

    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([concatenated_1, concatenated_2])

    model = Model([input_a, input_b, input_c, input_d], distance)
    model.summary()

    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model,base_network_1,base_network_2


def create_mul_combine_siamese_network(input_shape_1,input_shape_2):
    
    base_network_1 = mlp_network((60,1))
    base_network_2 = conv_network(input_shape_2)

    input_a = Input(shape=input_shape_1)
    input_b = Input(shape=input_shape_1)

    
    input_c = Input(shape=input_shape_2)
    input_d = Input(shape=input_shape_2)
    processed_c = base_network_2(input_c)
    processed_d = base_network_2(input_d)
    processed_c = Dense(30, activation='relu')(processed_c)
    processed_d = Dense(30, activation='relu')(processed_d)
    processed_c = Reshape((30,1))(processed_c)
    processed_d = Reshape((30,1))(processed_d)

    concatenated_1 = Concatenate()([input_a, processed_c])
    concatenated_2 = Concatenate()([input_b, processed_d])

    concatenated_1 = Reshape((60,1))(concatenated_1)
    concatenated_2 = Reshape((60,1))(concatenated_2)
    # x_1 = Dense(128, activation='relu')(concatenated_1)
    # x_2 = Dense(128, activation='relu')(concatenated_2)

    processed_a = base_network_1(concatenated_1)
    processed_b = base_network_1(concatenated_2)

    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b, input_c, input_d], distance)
    model.summary()
    base_network_1.summary()
    base_network_2.summary()

    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model,base_network_1,base_network_2


# def create_mul_combine_siamese_network_2(input_shape_1,input_shape_2):
    
#     base_network_1 = mlp_network(input_shape_1)
#     base_network_2 = conv_network(input_shape_2)

#     input_a = Input(shape=input_shape_1)
#     input_b = Input(shape=input_shape_1)
#     processed_a = base_network_1(input_a)
#     processed_b = base_network_1(input_b)
    
#     input_c = Input(shape=input_shape_2)
#     input_d = Input(shape=input_shape_2)
#     processed_c = base_network_2(input_c)
#     processed_d = base_network_2(input_d)

#     concatenated_1 = Concatenate()([processed_a, processed_c])
#     concatenated_2 = Concatenate()([processed_b, processed_d])

#     x_1 = Dense(128, activation='relu')(concatenated_1)
#     x_2 = Dense(128, activation='relu')(concatenated_2)

#     distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([concatenated_1, concatenated_2])

#     model = Model([input_a, input_b, input_c, input_d], distance)
#     model.summary()

#     rms = optimizers.RMSprop()
#     model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
#     # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
#     return model,base_network_1,base_network_2

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

