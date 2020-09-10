# -*- coding=utf-8 -*-


from sklearn.model_selection import train_test_split
from siamese.siamese_base import *
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

#fit_generator的迭代器，从内存中输入数据段到gpu显存中
def pairSequence(data,target,batch_size):
    datalen=len(data)
    print(datalen)
    print("迭代次数：",(int(datalen/batch_size)+1))
    while 1:  
        cnt = 0
        temppair=[]
        templabel=[]
        for i in range(0,datalen):
            temppair.append(data[i])
            templabel.append(target[i])
            cnt += 1
            if cnt==batch_size or i==(datalen-1):
                cnt = 0
                temppair=np.array(temppair)
                templabel=np.array(templabel)
                tempdata=[temppair[:, 0],temppair[:, 1]]
                yield(tempdata,templabel)      
                temppair=[]
                templabel=[]




def siamese_oridata(train_data,test_data, train_target, test_target,num_classes):

    # train_pairs, train_label = create_pairs(train_data, train_target,num_classes)
    train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
    # train_pairs, train_label = create_pairs_incre_2(train_data, digit_indices,num_classes)

    # test_pairs, test_label = create_pairs(test_data, digit_indices,num_classes)
    test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
    # test_pairs, test_label = create_pairs_incre_2(test_data, digit_indices,num_classes)
  
    print(train_pairs.shape)
    print(test_pairs.shape)

    input_shape = (len(train_data[0]),len(train_data[0][0]))
    # input_shape = (len(train_data[0]),len(train_data[0][0]),1)

	#配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
    # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, len(train_data[0]), len(train_data[0][0]), 1)  
    # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, len(train_data[0]), len(train_data[0][0]), 1) 
  
    model=create_siamese_network(input_shape)
    # model=create_presudo_siamese_network(input_shape)

    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=1024,epochs=100,
           validation_split=0.2)  

    # train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)
    # train_pairs=tf.cast(train_pairs,tf.float32)
    # val_pairs=tf.cast(val_pairs,tf.float32)
    # test_pairs=tf.cast(test_pairs,tf.float32)
    # train_label=tf.cast(train_label, tf.float32)
    # val_label=tf.cast(val_label, tf.float32)
    # test_label=tf.cast(test_label, tf.float32)
    # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
    #       batch_size=1024,epochs=100,
    #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    
    # 爆内存时使用
    # batch_size=1024
    # model.fit_generator(pairSequence(train_pairs, train_label, batch_size),
    #   epochs=50,steps_per_epoch=(int(len(train_pairs)/batch_size)+1)
    #   )

    model.save_weights('my_model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
    #计算训练集中的判定分数的均值
    interscore=[]
    intrascore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            interscore.append(train_pred[i])
        else:
            intrascore.append(train_pred[i])
    interscore=np.mean(interscore)
    intrascore=np.mean(intrascore)
    print("类间训练集分数：",interscore)
    print("类外训练集分数：",intrascore)

    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    tr_acc = compute_accuracy(train_label, train_pred)
    te_acc = compute_accuracy(test_label, test_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return test_pred,test_label

def siamese_cwt(train_data,test_data, train_target, test_target,num_classes):


    # train_pairs, train_label = create_pairs(train_data, digit_indices,num_classes)
    train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
    # train_pairs, train_label = create_pairs_incre_2(train_data, digit_indices,num_classes)

    # test_pairs, test_label = create_pairs(test_data, digit_indices,num_classes)
    test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
    # test_pairs, test_label = create_pairs_incre_2(test_data, digit_indices,num_classes)
    
    print(train_pairs.shape)
    print(test_pairs.shape)
    
    input_shape = (len(train_data[0]),len(train_data[0][0]),2)
    # input_shape = (2,24,300)
    # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 2, 64, 300)  #配对个数，每对有2组数据，每组数据有2个信号源，每个信号源有32维，300长
    # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 2, 64, 300) 
    model=create_siamese_network(input_shape)
    # model=create_presudo_siamese_network(input_shape)

    #配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
 
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    train_pairs = train_pairs.astype(np.float16)
    train_label = train_label.astype(np.float16)

    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=128,epochs=50,
           validation_split=0.2)  
    
    # train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)
    # train_pairs = train_pairs.astype(np.float16)
    # train_label = train_label.astype(np.float16)
    # val_pairs = val_pairs.astype(np.float16)
    # val_label = val_label.astype(np.float16)
    # test_pairs = test_pairs.astype(np.float16)
    # test_label = test_label.astype(np.float16)

    # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
    #       batch_size=128,epochs=50,
    #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    

    model.save_weights('my_model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])

    #计算训练集中的判定分数的均值
    interscore=[]
    intrascore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            interscore.append(train_pred[i])
        else:
            intrascore.append(train_pred[i])
    interscore=np.mean(interscore)
    intrascore=np.mean(intrascore)
    print("类间训练集分数：",interscore)
    print("类外训练集分数：",intrascore)

    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    tr_acc = compute_accuracy(train_label, train_pred)
    te_acc = compute_accuracy(test_label, test_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return test_pred,test_label





def siamese_feature(train_data,test_data, train_target, test_target,num_classes):

    train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
    # train_pairs, train_label = create_pairs_incre_2(train_data, train_target,num_classes)


    # test_pairs, test_label = create_pairs(test_data, test_target,num_classes)
    test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
    # test_pairs, test_label = create_pairs_incre_2(test_data, test_target,num_classes)
    # test_pairs, test_label = create_test_pair(test_data, test_target,num_classes)

    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)

    input_shape = (len(train_data[0]))
    # input_shape = (2,300,1)

    #配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
    # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 8, 300, 1)  
    # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 8, 300, 1) 

  
    model=create_siamese_network(input_shape)
    # model=create_presudo_siamese_network(input_shape)
    
    # train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)

    # train_pairs=tf.cast(train_pairs,tf.float32)
    # val_pairs=tf.cast(val_pairs,tf.float32)
    # test_pairs=tf.cast(test_pairs,tf.float32)
    # train_label=tf.cast(train_label, tf.float32)
    # val_label=tf.cast(val_label, tf.float32)
    # test_label=tf.cast(test_label, tf.float32)

    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=8192, epochs=200, 
           validation_split=0.2)  

    # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
    #       batch_size=128,
    #       epochs=50,verbose=2,
    #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    
    # batch_size=1024
    # model.fit_generator(pairSequence(train_pairs, train_label, batch_size),
    #   epochs=50,steps_per_epoch=(int(len(train_pairs)/batch_size)+1)
    #   )


    model.save_weights('my_model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
    #计算训练集中的判定分数的均值
    interscore=[]
    intrascore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            interscore.append(train_pred[i])
        else:
            intrascore.append(train_pred[i])
    interscore=np.mean(interscore)
    intrascore=np.mean(intrascore)
    print("类间训练集分数：",interscore)
    print("类外训练集分数：",intrascore)

    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    #对样本对进行额外处理
    # temp_pred=[]
    # for i in range(int(len(test_label))):
    #     temppred=0
    #     # temp_label.append(test_label[i*5])
    #     for j in range(5):
    #         temppred+=test_pred[i*5+j]
    #     temp_pred.append(temppred/5)
    # test_pred=temp_pred
    # print("len(test_pred):",len(test_pred))
    # print("len(test_label):",len(test_label))
    
    train_label=np.array(train_label)
    train_pred=np.array(train_pred)
    test_label=np.array(test_label)
    test_pred=np.array(test_pred)

    tr_acc = compute_accuracy(train_label, train_pred)
    te_acc = compute_accuracy(test_label, test_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return test_pred,test_label


def siamese_ori_final(data,target,num_classes):

    pairs, label = create_pairs_incre_1(data, target,num_classes)
    input_shape = (len(data[0]),len(data[0][0]))
   
    model=create_siamese_network(input_shape)
    model.load_weights('my_model_weights.h5')
    
    pred = model.predict([pairs[:, 0], pairs[:, 1]])
    return pred,label




def siamese_feature_buildmodel(train_data, train_target,num_classes):

    # train_pairs, train_label = create_pairs_incre_1(train_data, digit_indices,num_classes)
    train_pairs, train_label = create_pairs_incre_2(train_data, train_target,num_classes)

    print("train_pairs.shape:",train_pairs.shape)

    # input_shape = (1,len(train_data[0]))
    input_shape = (len(train_data[0]))

    #配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
    # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 8, 300, 1)  
    # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 8, 300, 1) 
  
    model=create_siamese_network(input_shape)
    # model=create_presudo_siamese_network(input_shape)
    
    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=8192,epochs=50)  

    # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
    #       batch_size=128,
    #       epochs=50,verbose=2,
    #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    
    # batch_size=1024
    # model.fit_generator(pairSequence(train_pairs, train_label, batch_size),
    #   epochs=50,steps_per_epoch=(int(len(train_pairs)/batch_size)+1)
    #   )

    model.save_weights('model_weights.h5')

    train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])

    #计算训练集中的判定分数的均值
    interscore=[]
    intrascore=[]
    for i in range(len(train_label)):
        if train_label[i]==1:
            interscore.append(train_pred[i])
        else:
            intrascore.append(train_pred[i])
    interscore=np.mean(interscore)
    intrascore=np.mean(intrascore)
    print("类间训练集分数：",interscore)
    print("类外训练集分数：",intrascore)

    tr_acc = compute_accuracy(train_label, train_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))




def siamese_feature_final(data,target,num_classes):
    # test_pairs, test_label = create_pairs_incre_1(data, target,num_classes)
    test_pairs, test_label = create_test_pair(data, target,num_classes)
    input_shape = (len(data[0]))
    print(input_shape)
   
    model=create_siamese_network(input_shape)
    model.load_weights('model_weights.h5')
    test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    #对样本对进行额外处理
    temp_pred=[]
    for i in range(int(len(test_label))):
        temppred=0
        # temp_label.append(test_label[i*5])
        for j in range(5):
            temppred+=test_pred[i*5+j]
        temp_pred.append(temppred/5)
    test_pred=temp_pred
    print("len(test_pred):",len(test_pred))
    print("len(test_label):",len(test_label))

    test_label=np.array(test_label)
    test_pred=np.array(test_pred)

    return test_pred,test_label

