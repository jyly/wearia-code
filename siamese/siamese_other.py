# -*- coding=utf-8 -*-
import os     
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ["PATH"] += os.pathsep + 'E:/system/python/graphviz/bin'
import tensorflow as tf
import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D
from keras import backend as K
from keras import regularizers,optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model,model_from_json
from sklearn.model_selection import train_test_split
from siamese import *

#参考文章的孪生网络方案


def emgauth_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(16, (2, 1), activation='relu', input_shape=input_shape)(input)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (1, 3), activation='relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)



def siamese_emgauth(train_data,test_data, train_target, test_target,num_classes):

    digit_indices = [np.where(train_target == i)[0] for i in range(1,num_classes+1)]
    train_pairs, train_label = create_pairs_incre(train_data, digit_indices,num_classes)

    digit_indices = [np.where(test_target == i)[0] for i in range(1,num_classes+1)]
    test_pairs, test_label = create_pairs_incre(test_data, digit_indices,num_classes)
  
    print(train_pairs.shape)
    print(test_pairs.shape)

    input_shape = (2,300,1)

	#配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
    train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 2, 300, 1)  
    test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 2, 300, 1) 

  
    base_network = emgauth_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    # keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
    model.summary()

    # train
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    
    train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)

    train_pairs=tf.cast(train_pairs,tf.float32)
    val_pairs=tf.cast(val_pairs,tf.float32)
    test_pairs=tf.cast(test_pairs,tf.float32)
    train_label=tf.cast(train_label, tf.float32)
    val_label=tf.cast(val_label, tf.float32)
    test_label=tf.cast(test_label, tf.float32)

    history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
          batch_size=128,
          epochs=50,verbose=2,
          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))

    model.save_weights('my_model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
    #计算训练集中的判定分数的均值
    allscore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            allscore.append(train_pred[i])
    allscore=np.mean(allscore)
    # allscore=1
    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    tr_acc = compute_accuracy(train_label, train_pred)
    te_acc = compute_accuracy(test_label, test_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return test_pred,test_label,allscore


def cwt_emg_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = MaxPooling2D(pool_size=(2, 4))(input)
    x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(2, 3))(input)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(input)
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)

    x = MaxPooling2D(pool_size=(2, 3))(input)
    x = Conv2D(128, (2, 6), activation='relu',padding='same')(x)
    # x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)

def siamese_cwt_emg(train_data,test_data, train_target, test_target,num_classes):

    digit_indices = [np.where(train_target == i)[0] for i in range(1,num_classes+1)]
    train_pairs, train_label = create_pairs(train_data, digit_indices,num_classes)
    # train_pairs, train_label = create_pairs_incre_1(train_data, digit_indices,num_classes)
    # train_pairs, train_label = create_pairs_incre(train_data, digit_indices,num_classes)

    digit_indices = [np.where(test_target == i)[0] for i in range(1,num_classes+1)]
    test_pairs, test_label = create_pairs(test_data, digit_indices,num_classes)
    # test_pairs, test_label = create_pairs_incre_1(test_data, digit_indices,num_classes)
    # test_pairs, test_label = create_pairs_incre(test_data, digit_indices,num_classes)
  
    print(train_pairs.shape)
    print(test_pairs.shape)

    input_shape = (48,300,2)
    # input_shape = (2,32,300)

	#配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
    train_pairs = train_pairs.reshape(train_pairs.shape[0], 2,48,300, 2)  
    test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 48,300, 2) 
    # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 2,32,300)  
    # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 2, 32,300) 
  
    base_network = cwt_emg_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    # keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
    model.summary()

    # train
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_2, optimizer=rms, metrics=[accuracy])

    
    train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)

    train_pairs=tf.cast(train_pairs,tf.float32)
    val_pairs=tf.cast(val_pairs,tf.float32)
    test_pairs=tf.cast(test_pairs,tf.float32)
    train_label=tf.cast(train_label, tf.float32)
    val_label=tf.cast(val_label, tf.float32)
    test_label=tf.cast(test_label, tf.float32)

    history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
          batch_size=32,
          epochs=50,verbose=2,
          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))

    model.save_weights('my_model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
    #计算训练集中的判定分数的均值
    allscore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            allscore.append(train_pred[i])
    allscore=np.mean(allscore)
    # allscore=1
    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    tr_acc = compute_accuracy(train_label, train_pred)
    te_acc = compute_accuracy(test_label, test_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return test_pred,test_label,allscore


