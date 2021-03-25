# -*- coding=utf-8 -*-
import numpy as np
import random
from tensorflow.keras import backend as K


#tensor里距离的计算方法
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


#两种对比损失的计算方法
def contrastive_loss_1(y_true, y_pred):#实际值，预测值,全部正确时返回1，全部错误时返回0
   # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def contrastive_loss_2(y_true, y_pred):#实际值，预测值,全部正确时返回1，全部错误时返回0
    Q=5
    sqaure_pred = K.square(y_pred)
    exp_pred=K.exp(-(float(2.77/Q)*y_pred))
    return K.mean((y_true)*float(2/Q)*sqaure_pred+(1-y_true)*2*Q*exp_pred)

#计算孪生网络的精度
def accuracy(y_true, y_pred): # Tensor上的操作
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


#fit_generator的迭代器，从内存中逐渐输入数据段到gpu显存中,避免炸显存
# def pairSequence(data,target,batch_size):
#     datalen=len(data)
#     print(datalen)
#     print("迭代次数：",(int(datalen/batch_size)+1))
#     while 1:  
#         cnt = 0
#         temppair=[]
#         templabel=[]
#         for i in range(0,datalen):
#             temppair.append(data[i])
#             templabel.append(target[i])
#             cnt += 1
#             if cnt==batch_size or i==(datalen-1):
#                 cnt = 0
#                 temppair=np.array(temppair)
#                 templabel=np.array(templabel)
#                 tempdata=[temppair[:, 0],temppair[:, 1]]
#                 yield(tempdata,templabel)      
#                 temppair=[]
#                 templabel=[]
#fit函数             
# model.fit_generator(pairSequence(train_pairs, train_label, batch_size),
#   epochs=50,steps_per_epoch=(int(len(train_pairs)/batch_size)+1)
#   )

#基础配对版本
def create_pairs_based(data, target,num_classes):
    #返回目标类别所在的序号构成的列表
    digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
    pairs = []
    labels = []
    for d in range(num_classes):
        n=len(digit_indices[d])-1
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[data[z1], data[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            inc_2 =random.randrange(0, len(digit_indices[dn]))
            z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
            pairs += [[data[z1], data[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#全内部样本,单敌对样本    
def create_pairs_incre(data, target,num_classes):
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
                inc_2 =random.randrange(0, len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                pairs += [[data[z1], data[z2]]]
                z1, z2 = digit_indices[d][j], digit_indices[dn][inc_2]
                pairs += [[data[z1], data[z2]]]
                labels += [1, 0, 0]
    return np.array(pairs), np.array(labels)




#pairs 是label的ancornum倍
def create_test_pair(test_data, test_target,num_classes,anchornum):
    # 为不同的类提取锚标点
    index=0
    tempdata=[[]]
    for i in range(len(test_target)):
        if test_target[i]==(index+1):
            tempdata[index].append(test_data[i])
        else:
            index=index+1
            tempdata.append([])

    pairs = []
    labels = []   
           
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
    #根据锚生产具体的配对
    for i in range(num_classes):
        for j in range(len(test_data[i])):
            # 添加合法样本对
            for k in range(anchornum):
                pairs+=[[test_data_anchor[i][k],test_data[i][j]]]
            labels += [1]
            #添加impost样本对
            for dk in range(2):
                inc_1 = random.randrange(1, num_classes)
                dn = (i + inc_1) % num_classes
                inc_2 =np.random.randint(0, len(tempdata[dn]))
                for k in range(anchornum):
                    pairs += [[test_data_anchor[i][k],tempdata[dn][inc_2]]]
                labels += [0]
    # print("len(test_pairs):",len(pairs))
    # print("len(test_labels):",len(labels))            
    return np.array(pairs), np.array下`(labels)

#对样本对进行额外处理,将测试数据的同锚的距离求均值
def repro_test_pred(test_pred,test_label,anchornum):
    temp_pred=[]
    for i in range(int(len(test_label))):
        temppred=0
        for j in range(anchornum):
            temppred+=test_pred[i*anchornum+j]
        temp_pred.append(temppred/anchornum)
    test_pred=temp_pred
    print("len(test_pred):",len(test_pred))
    print("len(test_label):",len(test_label))
    return test_pred



def dataresize(train_pairs,test_pairs,train_label,test_label):
    train_pairs = train_pairs.astype(np.float32)
    test_pairs = test_pairs.astype(np.float32)
    train_label = train_label.astype(np.float32)
    test_label = test_label.astype(np.float32)
    return train_pairs,test_pairs,train_label,test_label



#两种多任务的配对方案
# def create_multask_pair(data, target,num_classes):
#     digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
#     usernum=int(num_classes/9)
#     pairs = []
#     user_labels = []
#     gesture_labels = []
#     for k in range(usernum):
#         for t in range(9):
#             d=k*9+t
#             n=len(digit_indices[d])-1
#             for i in range(n):
#                 for j in range(i+1,n+1):
#                     z1, z2 = digit_indices[d][i], digit_indices[d][j]
#                     pairs += [[data[z1], data[z2]]]

#                     #不同用户同一手势的配对
#                     inc_1 = random.randrange(1, usernum)
#                     sn = (k+inc_1)%usernum
#                     gn = sn*9+t
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     pairs += [[data[z1], data[z2]]]
#                     z1, z2 = digit_indices[d][j], digit_indices[gn][inc_2]
#                     pairs += [[data[z1], data[z2]]]


#                     #同一用户不同手势的配对
#                     inc_1 = random.randrange(1, 9)
#                     dn = (t + inc_1) % 9
#                     gn = k*9 +dn
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     pairs += [[data[z1], data[z2]]]
#                     z1, z2 = digit_indices[d][j], digit_indices[gn][inc_2]
#                     pairs += [[data[z1], data[z2]]]


#                     #不同用户不同手势的配对
#                     gn = sn*9+dn
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     pairs += [[data[z1], data[z2]]]

#                     user_labels += [1, 1, 1,0, 0, 0]
#                     gesture_labels += [1, 0, 0, 1, 1, 0]
                    # labels=[user_labels,gesture_labels]
#     return np.array(pairs), np.array(labels)


# def create_multask_pair(data, target,num_classes):
#     digit_indices = [np.where(target == i)[0] for i in range(1,num_classes+1)]
#     usernum=int(num_classes/9)
#     user_pairs = []
#     gesture_pairs = []
#     user_labels = []
#     gesture_labels = []
#     for k in range(usernum):
#         for t in range(9):
#             d=k*9+t
#             n=len(digit_indices[d])-1
#             for i in range(n):
#                 for j in range(i+1,n+1):
#                     z1, z2 = digit_indices[d][i], digit_indices[d][j]
#                     user_pairs += [[data[z1], data[z2]]]
#                     gesture_pairs += [[data[z1], data[z2]]]

#                     #不同用户同一手势的配对
#                     inc_1 = random.randrange(1, usernum)
#                     sn = (k+inc_1)%usernum
#                     gn = sn*9+t
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     user_pairs += [[data[z1], data[z2]]]
#                     z1, z2 = digit_indices[d][j], digit_indices[gn][inc_2]
#                     user_pairs += [[data[z1], data[z2]]]

#                     #同一用户不同手势的配对
#                     inc_1 = random.randrange(1, 9)
#                     dn = (t + inc_1) % 9
#                     gn = k*9 +dn
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     gesture_pairs += [[data[z1], data[z2]]]
#                     z1, z2 = digit_indices[d][j], digit_indices[gn][inc_2]
#                     gesture_pairs += [[data[z1], data[z2]]]


#                     #不同用户不同手势的配对
#                     gn = sn*9+dn
#                     inc_2 =random.randrange(0, len(digit_indices[gn]))
#                     z1, z2 = digit_indices[d][i], digit_indices[gn][inc_2]
#                     user_pairs += [[data[z1], data[z2]]]
#                     gesture_pairs += [[data[z1], data[z2]]]

#                     user_labels += [1, 0, 0, 0]
#                     gesture_labels += [1, 0, 0, 0]


#                     pairs=[user_pairs,gesture_pairs]
#                     labels=[user_labels,gesture_labels]

#     return np.array(pairs),  np.array(labels)

#攻击测试实验配对
# def create_victima_test_pair(test_data, test_target,num_classes,anchornum):
#     tempdata=[[],[]]
#     for j in range(len(test_target)):
#         if test_target[j]==1:
#             tempdata[0].append(test_data[j])
#         else:
#             tempdata[1].append(test_data[j])
#     print("len(tempdata[0]):",len(tempdata[0]))
#     print("len(tempdata[1]):",len(tempdata[1]))

#     pairs = []
#     labels = []   
#     for t in range(3):            
#         test_data_anchor=[]
#         test_data=[]
#         #选择样本的锚和对比样本
#         rangek=list(range(len(tempdata[0])))
#         selectk = random.sample(rangek, anchornum)
#         for j in range(len(tempdata[0])):
#             if j in selectk:
#                 test_data_anchor.append(tempdata[0][j])
#             else:
#                 test_data.append(tempdata[0][j])
        
#         for j in range(len(test_data)):
#             # 添加合法样本对
#             for k in range(anchornum):
#                 pairs+=[[test_data_anchor[k],test_data[j]]]
#             labels += [1]
#             #添加impost样本对

#         resultList=random.sample(range(0,len(tempdata[1])),100);            

#         for t in resultList:
#             for k in range(anchornum):
#                 pairs += [[test_data_anchor[k],tempdata[1][t]]]
#             labels += [0]
#     # print("len(test_pairs):",len(pairs))
#     # print("len(test_labels):",len(labels))            
#     return np.array(pairs), np.array(labels)