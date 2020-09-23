# -*- coding=utf-8 -*-
import os     
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["PATH"] += os.pathsep + 'E:/system/python/graphviz/bin'
import tensorflow as tf
import keras
import numpy as np
import random
from keras.callbacks import TensorBoard
# from keras.datasets import manhattan_distance
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,MaxPooling1D,Conv1D,Reshape
from keras import backend as K
from keras import regularizers,optimizers
from keras.models import load_model,model_from_json
# import resnet_keras 

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
    # Q = tf.constant(Q, name="Q",dtype=tf.float32)
    # pos = tf.multiply(tf.multiply(1-y_true,2/Q),tf.square(y_pred))
    # neg = tf.multiply(tf.multiply(y_true,2*Q),tf.exp(-2.77/Q*y_pred))                
    # loss = pos + neg                 
    # loss = tf.reduce_mean(loss)              
    # return loss

def accuracy(y_true, y_pred): # Tensor上的操作

    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


#部分内部样本,单敌对样本  
def create_pairs(data, target,num_classes):
    #返回目标类别所在的序号构成的列表
    digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[data[z1], data[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[data[z1], data[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#全内部样本,单敌对样本    
def create_pairs_incre_1(data, target,num_classes):
    #返回目标类别所在的序号构成的列表
    digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
    pairs = []
    labels = []
    for d in range(num_classes):
        n=len(digit_indices[d])-1
        for i in range(n):
            for j in range(i+1,n+1):
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[data[z1], data[z2]]]
                inc_1 = random.randrange(1, num_classes)
                dn = (d + inc_1) % num_classes
                inc_2 =random.randrange(1, len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                pairs += [[data[z1], data[z2]]]
                labels += [1, 0]
    return np.array(pairs), np.array(labels)

#全内部样本,5敌对样本      
def create_pairs_incre_2(data, target,num_classes):
    pairs = []
    labels = []
    digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
    for d in range(num_classes):
        n=len(digit_indices[d])-1
        for i in range(n):
            for j in range(i+1,n+1):
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[data[z1], data[z2]]]
                labels += [1]
                for k in range(5):
                    inc_1 = random.randrange(1, num_classes)
                    dn = (d + inc_1) % num_classes
                    inc_2 =random.randrange(1, len(digit_indices[dn]))
                    z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                    pairs += [[data[z1], data[z2]]]
                    labels += [0]

    return np.array(pairs), np.array(labels)

def create_test_pair(test_data, test_target,num_classes,anchornum):
    tempdata=[]
    for i in range(1,num_classes+1):
        tempdata.append([])
        for j in range(len(test_target)):
            if test_target[j]==i:
                tempdata[i-1].append(test_data[j])
    pairs = []
    labels = []   
    for t in range(3):            
        test_data_anchor=[]
        test_data=[]
        #选择样本的锚和对比样本
        for i in range(num_classes):
            test_data_anchor.append([])
            test_data.append([])
            rangek=list(range(len(tempdata[i])))
            selectk = random.sample(rangek, anchornum)
            for j in range(len(tempdata[i])):
                if j in selectk:
                    test_data_anchor[i].append(tempdata[i][j])
                else:
                    test_data[i].append(tempdata[i][j])
        
        for i in range(num_classes):
            for j in range(len(test_data[i])):
                # 添加合法样本对
                for k in range(anchornum):
                    pairs+=[[test_data_anchor[i][k],test_data[i][j]]]
                labels += [1]
                #添加impost样本对
                for t in range(1):
                    inc_1 = random.randrange(1, num_classes)
                    dn = (i + inc_1) % num_classes
                    inc_2 =np.random.randint(0, len(tempdata[dn]))
                    for k in range(anchornum):
                        pairs += [[test_data_anchor[i][k],tempdata[dn][inc_2]]]
                    labels += [0]
    # print("len(test_pairs):",len(pairs))
    # print("len(test_labels):",len(labels))            
    return np.array(pairs), np.array(labels)




def mlp_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=(input_shape), name='input')
    # x = Flatten()(input)
    x = input
    #全连接层
    x = Dense(256, activation='relu')(x)
    #遗忘层
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
<<<<<<< HEAD
    x = Dense(256, activation='relu', name='output')(x)
=======
    x = Dense(256, activation='relu')(x)
>>>>>>> ec8e7f208371e0a041bef2dfaa4f72fb943391ca


    return Model(input, x)


def mlp_network_incre(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    #全连接层
    x = Dense(32, activation='relu')(x)
    #遗忘层
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)


def conv_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (2, 1), activation='relu', input_shape=input_shape,padding='same')(input)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (1, 3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def cwt_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
 
    # x = MaxPooling2D(pool_size=(2, 2))(input)
    x = Conv2D(16, (2, 2), activation='relu',padding='same')(input)
    x = Dropout(0.1)(x)

    # x = MaxPooling2D(pool_size=(2, 2))(input)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = Dropout(0.1)(x)

    # x = MaxPooling2D(pool_size=(2, 2))(input)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)

def vgg_16_base(input_shape):
    input = Input(shape=input_shape)

    net = Conv2D(64,(3,3),activation='relu',padding='same')(input)
    net = Conv2D(64,(3,3),activation='relu',padding='same')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
 
    net = Conv2D(128,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(128,(3,3),activation='relu',padding='same')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)

 
    net = Conv2D(256,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(256,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(256,(3,3),activation='relu',padding='same')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
 
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
 
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = Conv2D(512,(3,3),activation='relu',padding='same')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)

    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)

    return Model(input, net)

def create_siamese_network(input_shape):
    
    # base_network = conv_network(input_shape)
    base_network = mlp_network(input_shape)
    # base_network = mlp_network_incre(input_shape)
    # base_network = cwt_network(input_shape)
    # base_network = vgg_16_base(input_shape)
    # base_network = resnet_keras.ResNet50(input_shape)

    input_a = Input(shape=input_shape, name='input1')
    input_b = Input(shape=input_shape, name='input2')
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    x=Reshape((1,),name="output")(distance)
    model = Model([input_a, input_b], x)
    # keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
    model.summary()

    # train
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    return model


def create_siamese_network_2(input_shape):
    
    base_network = mlp_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    # x=Reshape((1,),name="output")(distance)
    # model = Model([input_a, input_b], x)
    model = Model([input_a, input_b], distance)
    model.summary()

    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss_1, optimizer=rms, metrics=[accuracy])
    return model,base_network


# def create_presudo_siamese_network(input_shape):
    
#     # base_network = conv_network(input_shape)
#     # base_network_1 = mlp_network(input_shape)
#     # base_network_2 = mlp_network_incre(input_shape)

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
