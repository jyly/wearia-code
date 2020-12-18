# -*- coding=utf-8 -*-
from sklearn.model_selection import train_test_split
from siamese.siamese_base import *
from siamese.siamese_tools import *
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from siamese.filecontrol import * 



def siamese_weighted_combine(train_data,test_data, train_target,test_target, trainindex,testindex,anchornum):
    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    traindatas_pairs=[]
    trainfeatures_pairs=[]
    for i in range(len(train_pairs)):
        traindatas_pairs.append([])
        trainfeatures_pairs.append([])
        for j in range(2):
            traindatas_pairs[i].append(train_pairs[i][j][0])
            trainfeatures_pairs[i].append(train_pairs[i][j][1])

    

    testdatas_pairs=[]
    testfeatures_pairs=[]
    for i in range(len(test_pairs)):
        testdatas_pairs.append([])
        testfeatures_pairs.append([])
        for j in range(2):
            testdatas_pairs[i].append(test_pairs[i][j][0])
            testfeatures_pairs[i].append(test_pairs[i][j][1])

    traindatas_pairs=np.array(traindatas_pairs)
    trainfeatures_pairs=np.array(trainfeatures_pairs)
    testdatas_pairs=np.array(testdatas_pairs)
    testfeatures_pairs=np.array(testfeatures_pairs)



    print("数据训练集对数：",traindatas_pairs.shape)
    print("特征训练集对数：",trainfeatures_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)
    print("特征测试集对数：",testfeatures_pairs.shape)
      
    # traindatas_pairs=traindatas_pairs.reshape(len(traindatas_pairs),2,len(traindatas_pairs[0][0]),len(traindatas_pairs[0][0][0]),1)
    # testdatas_pairs=testdatas_pairs.reshape(len(testdatas_pairs),2,len(testdatas_pairs[0][0]),len(testdatas_pairs[0][0][0]),1)

    print("数据训练集对数：",traindatas_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)

    trainfeatures_pairs=trainfeatures_pairs.reshape(len(trainfeatures_pairs),2,len(trainfeatures_pairs[0][0]),1)
    testfeatures_pairs=testfeatures_pairs.reshape(len(testfeatures_pairs),2,len(testfeatures_pairs[0][0]),1)
    print("特征训练集对数：",trainfeatures_pairs.shape)
    print("特征测试集对数：",testfeatures_pairs.shape)

    # ppg_data_shape = (2,len(traindatas_pairs[0][0][0]),1)
    # motion_data_shape = (6,len(traindatas_pairs[0][0][0]),1)
    ppg_data_shape = (len(train_data[0][0]),2)
    motion_data_shape = (len(train_data[0][0]),6)
    ppg_feature_shape = (30,1)
    motion_feature_shape = (30,1)
    print("ppg_data_shape:",ppg_data_shape)
    print("motion_data_shape:",motion_data_shape)
    print("ppg_feature_shape:",ppg_feature_shape)
    print("motion_feature_shape:",motion_feature_shape)

    ppg_data_model,ppg_data_based_model=create_siamese_network_lstm(ppg_data_shape)
    motion_data_model,motion_data_based_model=create_siamese_network_lstm(motion_data_shape)

    ppg_feature_model,ppg_feature_based_model=create_siamese_network_mlp(ppg_feature_shape)
    motion_feature_model,motion_feature_based_model=create_siamese_network_mlp(motion_feature_shape)


    ppg_data_model.fit([traindatas_pairs[:, 0,:,:2], traindatas_pairs[:, 1,:,:2]], train_label,  
       batch_size=512, epochs=10)  
    ppg_data_test_pred = ppg_data_model.predict([testdatas_pairs[:, 0,:,:2], testdatas_pairs[:, 1,:,:2]])

    motion_data_model.fit([traindatas_pairs[:, 0,:,2:], traindatas_pairs[:, 1,:,2:]], train_label,  
       batch_size=512, epochs=10)  
    motion_data_test_pred = motion_data_model.predict([testdatas_pairs[:, 0,:,2:], testdatas_pairs[:, 1,:,2:]])



    # ppg_data_model.fit([traindatas_pairs[:, 0,:2], traindatas_pairs[:, 1,:2]], train_label,  
    #    batch_size=512, epochs=10)  
    # ppg_data_test_pred = ppg_data_model.predict([testdatas_pairs[:, 0,:2], testdatas_pairs[:, 1,:2]])

    # motion_data_model.fit([traindatas_pairs[:, 0,2:], traindatas_pairs[:, 1,2:]], train_label,  
    #    batch_size=512, epochs=10)  
    # motion_data_test_pred = motion_data_model.predict([testdatas_pairs[:, 0,2:], testdatas_pairs[:, 1,2:]])

    ppg_feature_model.fit([trainfeatures_pairs[:, 0,:30], trainfeatures_pairs[:, 1,:30]], train_label,  
       batch_size=4096, epochs=40)  
    ppg_feature_test_pred = ppg_feature_model.predict([testfeatures_pairs[:, 0,:30], testfeatures_pairs[:, 1,:30]])

    motion_feature_model.fit([trainfeatures_pairs[:, 0,30:], trainfeatures_pairs[:, 1,30:]], train_label,  
       batch_size=4096, epochs=40)  
    motion_feature_test_pred = motion_feature_model.predict([testfeatures_pairs[:, 0,30:], testfeatures_pairs[:, 1,30:]])


    test_pred=[]
    for i in range(len(ppg_data_test_pred)):
        test_pred.append((ppg_data_test_pred[i]+motion_data_test_pred[i]+ppg_feature_test_pred[i]+motion_feature_test_pred[i])/4)
    # test_pred=ppg_test_pred
    return test_pred,test_label



def siamese_mul_combine(train_data,test_data, train_target,test_target, trainindex,testindex,anchornum):
    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    traindatas_pairs=[]
    trainfeatures_pairs=[]
    for i in range(len(train_pairs)):
        traindatas_pairs.append([])
        trainfeatures_pairs.append([])
        for j in range(2):
            traindatas_pairs[i].append(train_pairs[i][j][0])
            trainfeatures_pairs[i].append(train_pairs[i][j][1])

    testdatas_pairs=[]
    testfeatures_pairs=[]
    for i in range(len(test_pairs)):
        testdatas_pairs.append([])
        testfeatures_pairs.append([])
        for j in range(2):
            testdatas_pairs[i].append(test_pairs[i][j][0])
            testfeatures_pairs[i].append(test_pairs[i][j][1])

    traindatas_pairs=np.array(traindatas_pairs)
    trainfeatures_pairs=np.array(trainfeatures_pairs)
    testdatas_pairs=np.array(testdatas_pairs)
    testfeatures_pairs=np.array(testfeatures_pairs)
    print("数据训练集对数：",traindatas_pairs.shape)
    print("特征训练集对数：",trainfeatures_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)
    print("特征测试集对数：",testfeatures_pairs.shape)
      
    traindatas_pairs=traindatas_pairs.reshape(len(traindatas_pairs),2,len(traindatas_pairs[0][0]),len(traindatas_pairs[0][0][0]),1)
    testdatas_pairs=testdatas_pairs.reshape(len(testdatas_pairs),2,len(testdatas_pairs[0][0]),len(testdatas_pairs[0][0][0]),1)

    print("数据训练集对数：",traindatas_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)

    trainfeatures_pairs=trainfeatures_pairs.reshape(len(trainfeatures_pairs),2,len(trainfeatures_pairs[0][0]),1)
    testfeatures_pairs=testfeatures_pairs.reshape(len(testfeatures_pairs),2,len(testfeatures_pairs[0][0]),1)
    print("特征训练集对数：",trainfeatures_pairs.shape)
    print("特征测试集对数：",testfeatures_pairs.shape)

    ppg_data_shape = (2,len(traindatas_pairs[0][0][0]),1)
    motion_data_shape = (6,len(traindatas_pairs[0][0][0]),1)
    # ppg_data_shape = (len(train_data[0][0]),2)
    # motion_data_shape = (len(train_data[0][0]),6)
    ppg_feature_shape = (30,1)
    motion_feature_shape = (30,1)
    print("ppg_data_shape:",ppg_data_shape)
    print("motion_data_shape:",motion_data_shape)
    print("ppg_feature_shape:",ppg_feature_shape)
    print("motion_feature_shape:",motion_feature_shape)

    ppg_model,ppg_base_network_1,ppg_base_network_2=create_mul_combine_siamese_network(ppg_feature_shape,ppg_data_shape)
    motion_model,motion_base_network_1,motion_base_network_2=create_mul_combine_siamese_network(motion_feature_shape,motion_data_shape)

   
    ppg_model.fit([trainfeatures_pairs[:, 0,:30],trainfeatures_pairs[:, 1,:30],traindatas_pairs[:, 0,:2], traindatas_pairs[:, 1,:2]],
     train_label, batch_size=512, epochs=20)  
    ppg_pred = ppg_model.predict([testfeatures_pairs[:, 0,:30], testfeatures_pairs[:, 1,:30],testdatas_pairs[:, 0,:2], testdatas_pairs[:, 1,:2]])


    motion_model.fit([trainfeatures_pairs[:, 0,30:],trainfeatures_pairs[:, 1,30:],traindatas_pairs[:, 0,2:], traindatas_pairs[:, 1,2:]],
     train_label, batch_size=512, epochs=20)  
    motion_pred = motion_model.predict([testfeatures_pairs[:, 0,30:], testfeatures_pairs[:, 1,30:],testdatas_pairs[:, 0,2:], testdatas_pairs[:, 1,2:]])


    test_pred=[]
    for i in range(len(ppg_pred)):
        test_pred.append((ppg_pred[i]+motion_pred[i])/2)
    # test_pred=ppg_test_pred
    return test_pred,test_label


def siamese_mul_combine_2(train_data,test_data, train_target,test_target, trainindex,testindex,anchornum):
    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    traindatas_pairs=[]
    trainfeatures_pairs=[]
    for i in range(len(train_pairs)):
        traindatas_pairs.append([])
        trainfeatures_pairs.append([])
        for j in range(2):
            traindatas_pairs[i].append(train_pairs[i][j][0])
            trainfeatures_pairs[i].append(train_pairs[i][j][1])

    testdatas_pairs=[]
    testfeatures_pairs=[]
    for i in range(len(test_pairs)):
        testdatas_pairs.append([])
        testfeatures_pairs.append([])
        for j in range(2):
            testdatas_pairs[i].append(test_pairs[i][j][0])
            testfeatures_pairs[i].append(test_pairs[i][j][1])

    traindatas_pairs=np.array(traindatas_pairs)
    trainfeatures_pairs=np.array(trainfeatures_pairs)
    testdatas_pairs=np.array(testdatas_pairs)
    testfeatures_pairs=np.array(testfeatures_pairs)

    print("数据训练集对数：",traindatas_pairs.shape)
    print("特征训练集对数：",trainfeatures_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)
    print("特征测试集对数：",testfeatures_pairs.shape)
      
    traindatas_pairs=traindatas_pairs.reshape(len(traindatas_pairs),2,len(traindatas_pairs[0][0]),len(traindatas_pairs[0][0][0]),1)
    testdatas_pairs=testdatas_pairs.reshape(len(testdatas_pairs),2,len(testdatas_pairs[0][0]),len(testdatas_pairs[0][0][0]),1)

    print("数据训练集对数：",traindatas_pairs.shape)
    print("数据测试集对数：",testdatas_pairs.shape)



    ppg_data_shape = (2,len(traindatas_pairs[0][0][0]),1)
    motion_data_shape = (6,len(traindatas_pairs[0][0][0]),1)
    print("ppg_data_shape:",ppg_data_shape)
    print("motion_data_shape:",motion_data_shape)

    # ppg_data_model,ppg_data_based_model=create_siamese_network_lstm(ppg_data_shape)
    # motion_data_model,motion_data_based_model=create_siamese_network_lstm(motion_data_shape)
    
    ppg_data_model,ppg_data_based_model=create_siamese_network_conv(ppg_data_shape)
    motion_data_model,motion_data_based_model=create_siamese_network_conv(motion_data_shape)

    ppg_data_model.fit([traindatas_pairs[:, 0,:2], traindatas_pairs[:, 1,:2]], train_label,  
       batch_size=512, epochs=10)  
    motion_data_model.fit([traindatas_pairs[:, 0,2:], traindatas_pairs[:, 1,2:]], train_label,  
       batch_size=512, epochs=10)  

    print(traindatas_pairs[i][j][:2].shape)

    ppg_l=ppg_data_based_model.predict(traindatas_pairs[:,0,:2])
    ppg_r=ppg_data_based_model.predict(traindatas_pairs[:,1,:2])
    motion_l=motion_data_based_model.predict(traindatas_pairs[:,0,2:])
    motion_r=motion_data_based_model.predict(traindatas_pairs[:,1,2:])
    ppg_l=np.array(ppg_l)
    print(ppg_l.shape)
    ppg_train_fur_feature=[]
    motion_train_fur_feature=[]
    for i in range(len(traindatas_pairs)):
        ppg_pair=[]
        motion_pair=[]

        ppg_pred=ppg_l[i]
        ppg_pred=np.append(ppg_pred,trainfeatures_pairs[i][0][:30])
        ppg_pair.append(ppg_pred)
        ppg_pred=ppg_r[i]
        ppg_pred=np.append(ppg_pred,trainfeatures_pairs[i][1][:30])
        ppg_pair.append(ppg_pred)

        motion_pred=motion_l[i]
        motion_pred=np.append(motion_pred,trainfeatures_pairs[i][0][30:])
        motion_pair.append(motion_pred)
        motion_pred=motion_r[i]
        motion_pred=np.append(motion_pred,trainfeatures_pairs[i][1][30:])
        motion_pair.append(motion_pred)

        ppg_train_fur_feature.append(ppg_pair)
        motion_train_fur_feature.append(motion_pair)



    ppg_l=ppg_data_based_model.predict(testdatas_pairs[:,0,:2])
    ppg_r=ppg_data_based_model.predict(testdatas_pairs[:,1,:2])
    motion_l=motion_data_based_model.predict(testdatas_pairs[:,0,2:])
    motion_r=motion_data_based_model.predict(testdatas_pairs[:,1,2:])
    ppg_l=np.array(ppg_l)
    print(ppg_l.shape)

    ppg_test_fur_feature=[]
    motion_test_fur_feature=[]
    for i in range(len(testdatas_pairs)):
        ppg_pair=[]
        motion_pair=[]

        ppg_pred=ppg_l[i]
        ppg_pred=np.append(ppg_pred,trainfeatures_pairs[i][0][:30])
        ppg_pair.append(ppg_pred)
        ppg_pred=ppg_r[i]
        ppg_pred=np.append(ppg_pred,trainfeatures_pairs[i][1][:30])
        ppg_pair.append(ppg_pred)

        motion_pred=motion_l[i]
        motion_pred=np.append(motion_pred,trainfeatures_pairs[i][0][30:])
        motion_pair.append(motion_pred)
        motion_pred=motion_r[i]
        motion_pred=np.append(motion_pred,trainfeatures_pairs[i][1][30:])
        motion_pair.append(motion_pred)
        ppg_test_fur_feature.append(ppg_pair)
        motion_test_fur_feature.append(motion_pair)



    ppg_train_fur_feature=np.array(ppg_train_fur_feature)
    motion_train_fur_feature=np.array(motion_train_fur_feature)
    ppg_test_fur_feature=np.array(ppg_test_fur_feature)
    motion_test_fur_feature=np.array(motion_test_fur_feature)
    print("ppg_train_fur_feature:",ppg_train_fur_feature.shape)
    print("motion_train_fur_feature:",motion_train_fur_feature.shape)
    print("ppg_test_fur_feature:",ppg_test_fur_feature.shape)
    print("motion_test_fur_feature:",motion_test_fur_feature.shape)

    ppg_train_fur_feature=ppg_train_fur_feature.reshape(len(ppg_train_fur_feature),2,len(ppg_train_fur_feature[0][0]),1)
    motion_train_fur_feature=motion_train_fur_feature.reshape(len(motion_train_fur_feature),2,len(motion_train_fur_feature[0][0]),1)
    ppg_test_fur_feature=ppg_test_fur_feature.reshape(len(ppg_test_fur_feature),2,len(ppg_test_fur_feature[0][0]),1)
    motion_test_fur_feature=motion_test_fur_feature.reshape(len(motion_test_fur_feature),2,len(motion_test_fur_feature[0][0]),1)
 
    print("ppg_train_fur_feature:",ppg_train_fur_feature.shape)
    print("motion_train_fur_feature:",motion_train_fur_feature.shape)
    print("ppg_test_fur_feature:",ppg_test_fur_feature.shape)
    print("motion_test_fur_feature:",motion_test_fur_feature.shape)

    ppg_feature_shape = (60,1)
    motion_feature_shape = (60,1)
    print("ppg_feature_shape:",ppg_feature_shape)
    print("motion_feature_shape:",motion_feature_shape)

    ppg_feature_model,ppg_feature_based_model=create_siamese_network_mlp(ppg_feature_shape)
    motion_feature_model,motion_feature_based_model=create_siamese_network_mlp(motion_feature_shape)


    ppg_feature_model.fit([ppg_train_fur_feature[:, 0], ppg_train_fur_feature[:, 1]], train_label,  
       batch_size=4096, epochs=40)  
    ppg_feature_test_pred = ppg_feature_model.predict([ppg_test_fur_feature[:, 0], ppg_test_fur_feature[:, 1]])

    motion_feature_model.fit([motion_train_fur_feature[:, 0], motion_train_fur_feature[:, 1]], train_label,  
       batch_size=4096, epochs=40)  
    motion_feature_test_pred = motion_feature_model.predict([motion_test_fur_feature[:, 0], motion_test_fur_feature[:, 1]])


    test_pred=[]
    for i in range(len(ppg_feature_test_pred)):
        test_pred.append((ppg_feature_test_pred[i]+motion_feature_test_pred[i])/2)
    # test_pred=ppg_test_pred
    return test_pred,test_label