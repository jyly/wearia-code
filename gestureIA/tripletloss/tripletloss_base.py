import os
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os
from keras.datasets import mnist
from itertools import permutations
import random
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization,Conv2D, MaxPooling2D, ReLU,Dropout
from keras import regularizers,optimizers

def mlp(input_shape,num_classes):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    #全连接层
    x = Dense(128, activation='relu')(x)
    #遗忘层
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    pre_logit = Dense(128, activation='relu')(x)
    softmax = Dense(num_classes, activation='softmax')(pre_logit)
    return Model(inputs=[input], outputs=[softmax, pre_logit])


def mlp_network_incre(input_shape,num_classes):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    #全连接层
    x = Dense(16, activation='relu')(x)
    #遗忘层
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    pre_logit = Dense(128, activation='relu')(x)
    softmax = Dense(10, activation='softmax')(pre_logit)
    return Model(inputs=[input], outputs=[softmax, pre_logit])


def cnn(input,num_classes):
    net = Conv2D(2, kernel_size=(3, 3))(input)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Conv2D(4, (3, 3), strides=(2, 2))(net)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Flatten()(net)
    net = Dense(128)(net)
    net = BatchNormalization()(net)
    pre_logit = ReLU()(net)
    softmax = Dense(num_classes, activation='softmax')(pre_logit)
    return Model(inputs=[input], outputs=[softmax, pre_logits])


def generate_triplet(x, y,  ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])

    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        #构建积极对的全排列，再从全排列中随机选ap_pairs个出来
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]
            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test

    return np.array(triplet_train_pairs), np.array(y_triplet_pairs)

def generate_triplet_inc(x, y):
    num_classes=len(set(y))
    digit_indices = [np.where(y == i)[0] for i in range(0,num_classes)]
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z3 =  digit_indices[dn][i]
            pairs.append([x[z1], x[z2],x[z3]])
            labels.append([d,d,dn])

    return np.array(pairs), np.array(labels)

def triplet_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    # print('y_pred.shape = ', y_pred)
    # print('y_true.shape = ', y_true)
    
    alpha=0.4
    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    # pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=1))
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    # neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=1))
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, K.epsilon())

    return loss

def create_pairs(x, y,num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''

    digit_indices = [np.where(y == i)[0] for i in range(1,num_classes+1)]

    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

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


def create_tripletloss_network(input_shape,num_classes):
    # base_network = mlp(input_shape,num_classes)
    base_network = mlp_network_incre(input_shape,num_classes)

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')


    soft_anchor, pre_logits_anchor = base_network([anchor_input])
    soft_pos, pre_logits_pos = base_network([positive_input])
    soft_neg, pre_logits_neg = base_network([negative_input])

    merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')
    merged_pre = concatenate([pre_logits_anchor, pre_logits_pos, pre_logits_neg], axis=-1, name='merged_pre')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_soft, merged_pre])
    model.summary()
    rms = optimizers.RMSprop()
    model.compile(loss=["categorical_crossentropy", triplet_loss],optimizer=rms, metrics=["accuracy"])
    return model

