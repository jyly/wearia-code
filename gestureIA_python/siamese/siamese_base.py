# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from siamese.siamese_tools import *
from siamese.network import *
from tensorflow.keras.utils import  plot_model

import os
os.environ["PATH"] += ";E:/system/python/graphviz/bin/"




def create_siamese_network(input_shape,net_type):
    
    if net_type=='mlp':
        base_network=mlp_network(input_shape)
    if net_type=='lstm':
        base_network=lstm_network(input_shape)
    if net_type=='conv':
        base_network=conv_network(input_shape)
    if net_type=='conv_lstm':
        base_network=conv_lstm_network(input_shape)
    if net_type=='pyramid':
        base_network=pyramid_network(input_shape)
    if net_type=='bilstm':
        base_network=bilstm_network(input_shape)
    if net_type=='resnet1d':
        base_network=resnet1d_network(input_shape)


    base_network.summary()
    plot_model(base_network,"base_network.png",show_shapes=True)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    model.summary()

    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])

    # model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=[accuracy])
    # model.compile(loss='binary_crossentropy', optimizer=rms, metrics=[accuracy])
    # model.compile(loss='mean_squared_error', optimizer=rms, metrics=[accuracy])
    return model,base_network


def create_mul_task_siamese_network(input_shape):
    base_network=pyramid_network_based(input_shape)
    plot_model(base_network,"base_network.png",show_shapes=True)
    base_network.summary()
    #user user gesture gesture
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    input_c = Input(shape=input_shape)
    input_d = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    processed_c = base_network(input_c)
    processed_d = base_network(input_d)

    user_a=Dense(128, activation='relu')(processed_a)
    user_b=Dense(128, activation='relu')(processed_b)

    gesture_a=Dense(128, activation='relu')(processed_c)
    gesture_b=Dense(128, activation='relu')(processed_d)

    distance_user = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([user_a, user_b])
    distance_gesture = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([gesture_a, gesture_b])
    
    model = Model([input_a, input_b,input_c, input_d], [distance_user,distance_gesture])
    model.summary()

    rms = optimizers.RMSprop()
    # model.compile(loss={'distance_user':contrastive_loss_1,'distance_gesture':contrastive_loss_1},
    model.compile(loss=[contrastive_loss_1,contrastive_loss_1],loss_weights=[0.8, 0.2], optimizer=rms, metrics=[accuracy])
    return model,base_network



# def create_mul_task_siamese_network(input_shape):
#     base_network=pyramid_network_based(input_shape)
#     base_network.summary()
#     input_a = Input(shape=input_shape)
#     input_b = Input(shape=input_shape)
#     processed_a = base_network(input_a)
#     processed_b = base_network(input_b)

#     user_a=Dense(128, activation='relu')(processed_a)
#     gesture_a=Dense(128, activation='relu')(processed_a)


#     user_b=Dense(128, activation='relu')(processed_b)
#     gesture_b=Dense(128, activation='relu')(processed_b)

#     distance_user = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([user_a, user_b])
#     distance_gesture = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([gesture_a, gesture_b])
    
#     model = Model([input_a, input_b], [distance_user,distance_gesture])
#     model.summary()

#     rms = optimizers.RMSprop()
#     # model.compile(loss={'distance_user':contrastive_loss_1,'distance_gesture':contrastive_loss_1},
#     model.compile(loss=[contrastive_loss_1,contrastive_loss_1], optimizer=rms, metrics=[accuracy])
#     return model,base_network

# def create_mul_feature_siamese_network(input_shape_1,input_shape_2):
    
#     base_network_1 = mlp_network(input_shape_1)
#     base_network_2 = mlp_network(input_shape_2)

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

#     # averages=Concatenate()([x_1, x_2])
#     # distance = Dense(2, activation='softmax')(averages)
#     # distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([x_1, x_2])
#     distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([concatenated_1, concatenated_2])

#     model = Model([input_a, input_b, input_c, input_d], distance)
#     model.summary()
#     rms = optimizers.RMSprop()
#     model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
#     # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
#     return model,base_network_1,base_network_2


# def create_mul_data_siamese_network(input_shape_1,input_shape_2):
    
#     base_network_1 = conv_network(input_shape_1)
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


# def create_mul_combine_siamese_network(input_shape_1,input_shape_2):
    
#     base_network_1 = mlp_network((60,1))
#     base_network_2 = conv_network(input_shape_2)

#     input_a = Input(shape=input_shape_1)
#     input_b = Input(shape=input_shape_1)

    
#     input_c = Input(shape=input_shape_2)
#     input_d = Input(shape=input_shape_2)
#     processed_c = base_network_2(input_c)
#     processed_d = base_network_2(input_d)
#     processed_c = Dense(30, activation='relu')(processed_c)
#     processed_d = Dense(30, activation='relu')(processed_d)
#     processed_c = Reshape((30,1))(processed_c)
#     processed_d = Reshape((30,1))(processed_d)

#     concatenated_1 = Concatenate()([input_a, processed_c])
#     concatenated_2 = Concatenate()([input_b, processed_d])

#     concatenated_1 = Reshape((60,1))(concatenated_1)
#     concatenated_2 = Reshape((60,1))(concatenated_2)
#     # x_1 = Dense(128, activation='relu')(concatenated_1)
#     # x_2 = Dense(128, activation='relu')(concatenated_2)

#     processed_a = base_network_1(concatenated_1)
#     processed_b = base_network_1(concatenated_2)

#     distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

#     model = Model([input_a, input_b, input_c, input_d], distance)
#     model.summary()
#     base_network_1.summary()
#     base_network_2.summary()

#     rms = optimizers.RMSprop()
#     model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
#     # model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
#     return model,base_network_1,base_network_2


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


