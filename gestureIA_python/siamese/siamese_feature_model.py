# -*- coding=utf-8 -*-
from sklearn.model_selection import train_test_split
from siamese.siamese_base import *
from siamese.siamese_tools import *
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from siamese.filecontrol import * 

def siamese_feature(train_data,test_data, train_target,test_target, trainindex,testindex,anchornum):

    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)

    train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),1)
    test_pairs=test_pairs.reshape(len(test_pairs),2,len(test_data[0]),1)
    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    input_shape = (len(train_data[0]),1)
    print(input_shape)

    model,based_model=create_siamese_network_mlp(input_shape)
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=512, epochs=10)  
    #计算训练集中的判定分数的均值
    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    test_pred=np.array(test_pred)
    return test_pred,test_label

def siamese_weighted_feature(train_data,test_data, train_target,test_target, trainindex,testindex,featurenum,anchornum):

    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)
   
    train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),1)
    test_pairs=test_pairs.reshape(len(test_pairs),2,len(test_data[0]),1)
    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    input_shape = (featurenum,1)
    print(input_shape)

    ppg_model,ppg_based_model=create_siamese_network_mlp(input_shape)
    history = ppg_model.fit([train_pairs[:, 0,:featurenum], train_pairs[:, 1,:featurenum]], train_label,  
           batch_size=512, epochs=10)  
    ppg_test_pred = ppg_model.predict([test_pairs[:, 0,:featurenum], test_pairs[:, 1,:featurenum]])
    ppg_test_pred=np.array(ppg_test_pred)
   

    motion_model,motion_based_model=create_siamese_network_mlp(input_shape)
    history = motion_model.fit([train_pairs[:, 0,featurenum:], train_pairs[:, 1,featurenum:]], train_label,  
           batch_size=512, epochs=10)  
    motion_test_pred = motion_model.predict([test_pairs[:, 0,featurenum:], test_pairs[:, 1,featurenum:]])
    motion_test_pred=np.array(motion_test_pred)

    r=0.5
    test_pred=[]
    for i in range(len(ppg_test_pred)):
        # test_pred.append((ppg_test_pred[i]+motion_test_pred[i])/2)
        test_pred.append(r*ppg_test_pred[i]+(1-r)*motion_test_pred[i])
    # test_pred=ppg_test_pred
    return test_pred,test_label



def siamese_mul_model_feature(train_data,test_data, train_target,test_target, trainindex,testindex,featurenum,anchornum):

    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
   
    train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),1)
    test_pairs=test_pairs.reshape(len(test_pairs),2,len(test_data[0]),1)

    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)

    input_shape = (featurenum,1)

    mul_model,base_model_1,base_network_2=create_mul_feature_siamese_network(input_shape,input_shape)

    history = mul_model.fit([train_pairs[:, 0,:featurenum], train_pairs[:, 1,:featurenum], train_pairs[:, 0,featurenum:], train_pairs[:, 1,featurenum:]],
        train_label, batch_size=512, epochs=10)
    test_pred = mul_model.predict([test_pairs[:, 0,:featurenum], test_pairs[:, 1,:featurenum],test_pairs[:, 0,featurenum:], test_pairs[:, 1,featurenum:]])
    test_pred=np.array(test_pred)

    return test_pred,test_label






def siamese_feature_buildmodel(train_data, train_target,num_classes):
    # train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
    train_pairs, train_label = create_pairs_incre_2(train_data, train_target,num_classes)
    print("train_pairs.shape:",train_pairs.shape)
    train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),1)
    input_shape = (len(train_data[0]),1)

    model,based_model=create_siamese_network(input_shape)
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=8192,epochs=40,
           validation_split=0.2)  
    model.save_weights('./parameter/model_weights.h5')
    based_model.save_weights('./parameter/based_model_weights.h5')
    based_model.save('./parameter/based_model.h5')



def siamese_feature_final(data,target,num_classes,anchornum=5):
    test_pairs, test_label = create_pairs_incre_1(data, target,num_classes)
    # based_model=load_model('based_model.h5')
    # based_model.summary()
    # x=based_model.predict(data)
    # featurewrite2(x,target)

    # test_pairs, test_label = create_test_pair(data, target,num_classes,anchornum)
    input_shape = (len(data[0]),1)
    test_pairs=test_pairs.reshape(len(test_pairs),2,len(data[0]),1)

    print(input_shape)
   
    model,based_model=create_siamese_network(input_shape)
    model.load_weights('./parameter/model_weights.h5')

    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    print("len(test_pred):",len(test_pred))

    converter = tf.lite.TFLiteConverter.from_keras_model_file('./parameter/based_model.h5')
    tflite_model = converter.convert()
    open("./parameter/based_model.tflite", "wb").write(tflite_model)
    print("len(test_pred):",len(test_pred))
    print("len(test_label):",len(test_label))
    test_label=np.array(test_label)
    test_pred=np.array(test_pred)
    return test_pred,test_label




def siamese_mul_feature_buildmodel(train_data, train_target,num_classes,featurenum):
    # train_pairs, train_label = create_pairs_incre_1(train_data, digit_indices,num_classes)
    train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)

    train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),1)
    input_shape = (featurenum,1)
    print("train_pairs.shape:",train_pairs.shape)
    print(input_shape)
    
    # train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    ppg_model,ppg_based_model=create_siamese_network(input_shape)
    history = ppg_model.fit([train_pairs[:, 0,:featurenum], train_pairs[:, 1,:featurenum]], train_label,  
           batch_size=8192, epochs=200)  
    ppg_model.save_weights('./parameter/ppg_model_weights.h5')#原型因有lamdba层，不能直接保存模型
    ppg_based_model.save_weights('./parameter/ppg_based_model_weights.h5')
    ppg_based_model.save('./parameter/ppg_based_model.h5')
  
    input_shape = (len(train_data[0])-featurenum,1)
    motion_model,motion_based_model=create_siamese_network(input_shape)
    history = motion_model.fit([train_pairs[:, 0,featurenum:], train_pairs[:, 1,featurenum:]], train_label,  
           batch_size=8192, epochs=200)  
    motion_model.save_weights('./parameter/motion_model_weights.h5')
    motion_based_model.save_weights('./parameter/motion_based_model_weights.h5')
    motion_based_model.save('./parameter/motion_based_model.h5')

def siamese_mul_feature_final(test_data,test_target,num_classes,featurenum,anchornum=5):
    # test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
    test_pairs, test_label = create_test_pair(test_data, test_target,num_classes,anchornum)
    # test_pairs, test_label = create_victima_test_pair(test_data, test_target,num_classes,anchornum)
   
    test_pairs=test_pairs.reshape(len(test_pairs),2,len(test_data[0]),1)
    input_shape = (featurenum,1)
    print("test_pairs.shape:",test_pairs.shape)
    print(input_shape)

    ppg_model,ppg_based_network=create_siamese_network(input_shape)
    ppg_model.load_weights('./parameter/ppg_model_weights.h5')
   
    input_shape = (len(test_data[0])-featurenum,1)
    motion_model,motion_based_network=create_siamese_network(input_shape)
    motion_model.load_weights('./parameter/motion_model_weights.h5')
    ppg_test_pred = ppg_model.predict([test_pairs[:, 0,:featurenum], test_pairs[:, 1,:featurenum]])
    motion_test_pred = motion_model.predict([test_pairs[:, 0,featurenum:], test_pairs[:, 1,featurenum:]])


    # ppg_based_network =load_model('./parameter/ppg_based_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model_file('./parameter/ppg_based_model.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model(ppg_based_network)
    tflite_model = converter.convert()
    open("./parameter/ppg_based_model.tflite", "wb").write(tflite_model)
    # motion_based_network =load_model('./parameter/motion_based_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model_file('./parameter/motion_based_model.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model(motion_based_network)
    tflite_model = converter.convert()
    open("./parameter/motion_based_model.tflite", "wb").write(tflite_model)


    #对样本对进行额外处理
    temp_pred=[]
    for i in range(int(len(test_label))):
        temppred=0
        for j in range(anchornum):
            temppred+=ppg_test_pred[i*anchornum+j]
        temp_pred.append(temppred/anchornum)
    ppg_test_pred=temp_pred

    temp_pred=[]
    for i in range(int(len(test_label))):
        temppred=0
        for j in range(anchornum):
            temppred+=motion_test_pred[i*anchornum+j]
        temp_pred.append(temppred/anchornum)
    motion_test_pred=temp_pred

    print("len(ppg_test_pred):",len(ppg_test_pred))
    print("len(motion_test_pred):",len(motion_test_pred))

    test_pred=[]
    for i in range(len(ppg_test_pred)):
        test_pred.append((ppg_test_pred[i]+motion_test_pred[i])/2)
    test_pred=ppg_test_pred

    test_label=np.array(test_label)
    test_pred=np.array(test_pred)
    return test_pred,test_label

def siamese_mul_feature_single(train_data,test_data,featurenum):
    test_pairs = create_single_test_pair(train_data, test_data)
    input_shape = (featurenum,)
    print(input_shape)
    print(test_pairs.shape)

    ppg_model,ppg_based_network=create_siamese_network(input_shape)
    ppg_model.load_weights('ppg_model_weights.h5')

    motion_model,motion_based_network=create_siamese_network(input_shape)
    motion_model.load_weights('motion_model_weights.h5')

    ppg_test_pred = ppg_model.predict([test_pairs[:, 0,:featurenum], test_pairs[:, 1,:featurenum]])
    motion_test_pred = motion_model.predict([test_pairs[:, 0,featurenum:2*featurenum], test_pairs[:, 1,featurenum:2*featurenum]])

    # test_pred=[]
    # for i in range(len(ppg_test_pred)):
    #     test_pred.append((ppg_test_pred[i]+motion_test_pred[i])/2)

    test_pred=ppg_test_pred

    test_pred=np.array(test_pred)


    return test_pred


# def siamese_cwt(train_data,test_data, train_target, test_target,num_classes):


#     # train_pairs, train_label = create_pairs(train_data, digit_indices,num_classes)
#     train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
#     # train_pairs, train_label = create_pairs_incre_2(train_data, digit_indices,num_classes)

#     # test_pairs, test_label = create_pairs(test_data, digit_indices,num_classes)
#     test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
#     # test_pairs, test_label = create_pairs_incre_2(test_data, digit_indices,num_classes)
    
#     print(train_pairs.shape)
#     print(test_pairs.shape)
    
#     input_shape = (len(train_data[0]),len(train_data[0][0]),2)
#     # input_shape = (2,24,300)
#     # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 2, 64, 300)  #配对个数，每对有2组数据，每组数据有2个信号源，每个信号源有32维，300长
#     # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 2, 64, 300) 
#     model=create_siamese_network(input_shape)
#     # model=create_presudo_siamese_network(input_shape)

#     #配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
 
#     train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
#     train_pairs = train_pairs.astype(np.float16)
#     train_label = train_label.astype(np.float16)

#     history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
#            batch_size=128,epochs=50,
#            validation_split=0.2)  
    
#     # train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)
#     # train_pairs = train_pairs.astype(np.float16)
#     # train_label = train_label.astype(np.float16)
#     # val_pairs = val_pairs.astype(np.float16)
#     # val_label = val_label.astype(np.float16)
#     # test_pairs = test_pairs.astype(np.float16)
#     # test_label = test_label.astype(np.float16)

#     # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
#     #       batch_size=128,epochs=50,
#     #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    

#     model.save_weights('model_weights.h5')

#     train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])

#     #计算训练集中的判定分数的均值
#     interscore=[]
#     intrascore=[]
#     for i in range(len(train_label)):
#         if train_label[i]==1:
#             interscore.append(train_pred[i])
#         else:
#             intrascore.append(train_pred[i])
#     interscore=np.mean(interscore)
#     intrascore=np.mean(intrascore)
#     print("类间训练集分数：",interscore)
#     print("类外训练集分数：",intrascore)

#     test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

#     tr_acc = compute_accuracy(train_label, train_pred)
#     te_acc = compute_accuracy(test_label, test_pred)
#     print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#     print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
#     return test_pred,test_label


# def siamese_ori_final(data,target,num_classes):

#     pairs, label = create_pairs_incre_1(data, target,num_classes)
#     input_shape = (len(data[0]),len(data[0][0]))
   
#     model=create_siamese_network(input_shape)
#     model.load_weights('model_weights.h5')
    
#     pred = model.predict([pairs[:, 0], pairs[:, 1]])
#     return pred,label


#自构建架构
 # base_network.summary()
    # x=base_network.predict(data[:,:featurenum])
    # featurewrite2(x,target)

    # 自架构基础模型
    # input_a = Input(shape=input_shape)
    # input_b = Input(shape=input_shape)
    # processed_a = base_network(input_a)
    # processed_b = base_network(input_b)
    # distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    # ppg_model = Model([input_a, input_b], distance)

    # x=base_network.predict(test_pairs[:, 0,:featurenum])
    # y=base_network.predict(test_pairs[:, 1,:featurenum])
    # print(len(x[0]))
    # score=[]
    # for i in range(len(x)):
    #     temp=0
    #     for j in range(len(x[0])):
    #         temp=temp+(x[i][j]-y[i][j])*(x[i][j]-y[i][j])
    #     score.append(temp**0.5)
    # print(score)
    # ppg_test_pred=score