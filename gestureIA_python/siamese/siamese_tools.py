# -*- coding=utf-8 -*-
import numpy as np

import random

#部分内部样本,单敌对样本  
def create_pairs_based(data, target,num_classes):
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
            inc_2 =random.randrange(0, len(digit_indices[dn]))
            z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
            pairs += [[data[z1], data[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_pairs_based_2(data, target,num_classes):
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

def create_pairs_based_3(data, target,num_classes):
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

            for t in range(10):
                inc = random.randrange(1, n)
                ds=(i + inc) %  n
                z1, z2 = digit_indices[d][i], digit_indices[d][ds]
                pairs += [[data[z1], data[z2]]]

                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                inc_2 =random.randrange(0, len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
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
                inc_2 =random.randrange(0, len(digit_indices[dn]))
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
                for k in range(2):
                    inc_1 = random.randrange(1, num_classes)
                    dn = (d + inc_1) % num_classes
                    inc_2 =random.randrange(0, len(digit_indices[dn]))
                    z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                    pairs += [[data[z1], data[z2]]]
                    labels += [0]
                # for iters in range(5):
                for dk in range(3):
                    dn = (d + dk+1) % num_classes
                    inc_2 =random.randrange(0, len(digit_indices[dn]))
                    z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                    pairs += [[data[z1], data[z2]]]
                    dn = (d - dk-1) % num_classes
                    inc_2 =random.randrange(0, len(digit_indices[dn]))
                    z1, z2 = digit_indices[d][i], digit_indices[dn][inc_2]
                    pairs += [[data[z1], data[z2]]]
                    labels += [0,0]
    return np.array(pairs), np.array(labels)


#pairs 是label的ancornum倍
def create_test_pair(test_data, test_target,num_classes,anchornum):
    tempdata=[]
    for i in range(1,num_classes+1):
        tempdata.append([])
        for j in range(len(test_target)):
            if test_target[j]==i:
                tempdata[i-1].append(test_data[j])
    pairs = []
    labels = []   
    for t in range(1):            
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
                for dk in range(3):
                    # inc_1 = random.randrange(1, 6)
                    dn = (i + dk+1) % num_classes
                    inc_2 =np.random.randint(0, len(tempdata[dn]))
                    for k in range(anchornum):
                        pairs += [[test_data_anchor[i][k],tempdata[dn][inc_2]]]
                    dn = (i - dk-1) % num_classes
                    inc_2 =np.random.randint(0, len(tempdata[dn]))
                    for k in range(anchornum):
                        pairs += [[test_data_anchor[i][k],tempdata[dn][inc_2]]]
                    labels += [0,0]
    # print("len(test_pairs):",len(pairs))
    # print("len(test_labels):",len(labels))            
    return np.array(pairs), np.array(labels)

def create_single_test_pair(train_data, test_data):
    pairs=[]
    for i in range(len(test_data)):
        for j in range(len(train_data)):
            pairs+=[[test_data[i],train_data[j]]]
    return np.array(pairs)



def create_victima_test_pair(test_data, test_target,num_classes,anchornum):
    tempdata=[[],[]]
    for j in range(len(test_target)):
        if test_target[j]==1:
            tempdata[0].append(test_data[j])
        else:
            tempdata[1].append(test_data[j])
    print("len(tempdata[0]):",len(tempdata[0]))
    print("len(tempdata[1]):",len(tempdata[1]))

    pairs = []
    labels = []   
    for t in range(3):            
        test_data_anchor=[]
        test_data=[]
        #选择样本的锚和对比样本
        rangek=list(range(len(tempdata[0])))
        selectk = random.sample(rangek, anchornum)
        for j in range(len(tempdata[0])):
            if j in selectk:
                test_data_anchor.append(tempdata[0][j])
            else:
                test_data.append(tempdata[0][j])
        
        for j in range(len(test_data)):
            # 添加合法样本对
            for k in range(anchornum):
                pairs+=[[test_data_anchor[k],test_data[j]]]
            labels += [1]
            #添加impost样本对

        resultList=random.sample(range(0,len(tempdata[1])),100);            

        for t in resultList:
            for k in range(anchornum):
                pairs += [[test_data_anchor[k],tempdata[1][t]]]
            labels += [0]
    # print("len(test_pairs):",len(pairs))
    # print("len(test_labels):",len(labels))            
    return np.array(pairs), np.array(labels)

