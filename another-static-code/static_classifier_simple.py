import os     
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ["PATH"] += os.pathsep + 'E:/system/python/graphviz/bin'

import numpy as np
import matplotlib.pyplot as plt
import random
from minepy import MINE
# import tensorflow as tf
# import keras
# from keras.callbacks import TensorBoard
# from keras.models import Model
# from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,LeakyReLU
# from keras.optimizers import RMSprop
# from keras import backend as K
# from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


def filterparameterwrite(sort,lda_bar,lda_scaling,filename):
    
    outputfile=open(filename,'w+')
    outputfile.write(str(sort))
    outputfile.write('\n')
    outputfile.write(str(lda_bar))
    outputfile.write('\n')
    outputfile.write(str(lda_scaling))
    outputfile.close()


def filterparameterread(filename):
    
    inputfile=open(filename,'r+')
    parameter=[]
    for i in inputfile:
        i=list(eval(i))
        parameter.append(i)
    inputfile.close()   
    sort=parameter[0]
    lda_bar=parameter[1]
    lda_scaling=parameter[2]
    return sort,lda_bar,lda_scaling


# 线性判别式做特征降维
def ldapro(train_data,test_data,train_target):
    lda = LinearDiscriminantAnalysis()
    lda = lda.fit(train_data, train_target)
    train_data=lda.transform(train_data)
    # print(np.dot(test_data[0]-lda.xbar_,lda.scalings_))
    #降维等于 np.dot(test_data-lda_bar,lda_scaling)
    if len(test_data)>0:
        test_data=lda.transform(test_data)  
    # print(test_data[0])
    lda_bar=[i for i in lda.xbar_]
    print("lda_bar:",lda_bar)
    lda_scaling=[[j for j in i] for i in lda.scalings_]
    print("lda_scaling:",lda_scaling)
    return train_data,test_data,lda_bar,lda_scaling

#根据分数选择分数最高的前topn个特征
def scoreselect(data,sort,topn):
    if len(data)==0:
        return data
    if topn>len(data[0]):
        topn=len(data[0])
    selectdata=[]
    for i in range(len(data)):
        selectdata.append([])
        for j in range(topn):
            selectdata[i].append(data[i][sort[j]])
    return selectdata

#计算多个特征的信息熵
def minecal(data,target):
    informscore=[]
    for i in range(len(data[0])):
        tempdata=[]
        for j in range(len(data)):
            tempdata.append(data[j][i])
        mine = MINE()
        mine.compute_score(tempdata, target)
        informscore.append(mine.mic())
    informscore=np.array(informscore)
    informsort = np.argsort(-informscore)#由大到小排序，得到对应的序号
    temp=[i for i in informsort]
    informsort=temp
    print("informscore:",informscore)
    print("informsort:",informsort)
    return informsort


#根据序列，计算信息熵，返回最高信息量的前i个特征列
def minepro(train_data,test_data,train_target,i):#训练集数据，测试集数据，训练集结果，降维后的特征数
    sort=minecal(train_data,train_target)
    train_data=scoreselect(train_data,sort,i)
    test_data=scoreselect(test_data,sort,i)
    return train_data,test_data,sort

def stdpro(train_data,test_data):
    scaler = StandardScaler()
    scaler = scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    # print(np.dot(test_data[0]-lda.xbar_,lda.scalings_))
    #降维等于 np.dot(test_data-lda_bar,lda_scaling)
    if len(test_data)>0:
        test_data=scaler.transform(test_data)   
    # print(test_data[0])
    scaler_mean=[i for i in scaler.mean_]
    print("scaler_mean:",scaler_mean)
    scaler_scale=[i for i in scaler.scale_]
    print("scaler_scale:",scaler_scale)
    return train_data,test_data,scaler_mean,scaler_scale

def mlp_accuracy_result(target,result,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0

    for i in range(0,divnum):
        for j in range(len(result)):
            if np.argmax(result[j])==i:
                if target[j]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn

def mlp_accuracy_score(target,score,threshold,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0

    for i in range(0,divnum):
        for j in range(len(score)):
            if score[j][i]>threshold:
                if target[j]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn


def svm_accuracy_result(target,result,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0

    for i in range(0,divnum):
        for j in range(len(result)):
            if result[j]==i:
                if target[j]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn
def svm_accuracy_score(target,score,threshold,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(0,divnum):
        for j in range(len(score)):
            if score[j][i]>threshold:
                if target[j]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn





def simplemoel_mlp(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(38,activation='sigmoid')(x)

    return Model(input, x)

def loaddata():
    filepath="features-FTA.csv"
    data=[]
    target=[]
    inputfile=open(filepath,'r+')
    indexuser=-2
    indexusertarget=0
    print(filepath)
    for i in inputfile:
        if indexuser==-2:
            indexuser=indexuser+1
            continue
        t=eval(i[:-4])
        t=list(t)
        temp=[]
        for j in range(17):
            temp.append(t[j])
        data.append(temp)

        if indexusertarget!=i[-4:]:
            indexusertarget=i[-4:]
            indexuser=indexuser+1

        target.append(indexuser)
    inputfile.close()  
    return data,target 
# the data, split between train and test sets



def loaddata_2():
    data=[]
    target=[]
    filepath="f_cheating-FTA.csv"
    df = pd.read_csv(filepath, index_col=False)
    df1 = df.values.tolist()   
    filepath="f_fft-FTA.csv"
    df = pd.read_csv(filepath, index_col=False)
    df2 = df.values.tolist() 
    filepath="f_fiducial_points-FTA.csv"
    df = pd.read_csv(filepath, index_col=False)
    df3 = df.values.tolist() 
    filepath="f_widths-FTA.csv"
    df = pd.read_csv(filepath, index_col=False)
    df4 = df.values.tolist()
    userindex=1
    usertarget=0
    for i in range(len(df1)):
        temp=[]
        temp=temp+df1[i][3:]
        temp=temp+df2[i][3:]
        temp=temp+df3[i][3:]
        temp=temp+df4[i][3:]
        data.append(temp)

        if i==0:
            usertarget=df1[i][1]
        else:
            if usertarget!=df1[i][1]:
                userindex=userindex+1
                usertarget=df1[i][1]
        target.append(userindex)

    return data,target






def sklearn_svmclass(featureset,target,divnum):
    print(len(target))
    meanacc=[]
    meanfar=[]
    meanfrr=[]  
    for t in range(0,10):
        train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.3,random_state = t*30,stratify=target)
        print("进入第",t,"轮分类阶段")
        # train_data,test_data,sort=minepro(train_data,test_data,train_target,10)
        # train_data,test_data,lda_bar,lda_scaling=ldapro(train_data,test_data,train_target)
        # filterparameterwrite(sort,lda_bar,lda_scaling,'./ldapropara.txt')


        sort,scale_mean,scale_scale=filterparameterread('./ldapropara.txt')
        # scale_mean=np.array(scale_mean)
        # scale_scale=np.array(scale_scale)
        train_data=scoreselect(train_data,sort,10)
        test_data=scoreselect(test_data,sort,10)
        # train_data=(train_data-scale_mean)/scale_scale
        # train_data,test_data,lda_bar,lda_scaling=ldapro(train_data,test_data,train_target)


        clf = SVC(probability=True)

        clf.fit(X=train_data, y=train_target)

        result = clf.predict(test_data)
        score = clf.predict_proba(test_data)
        print('原结果：',test_target)
        print('预测结果：',result)
        print('预测分数：',score)
        # tp,tn,fp,fn=svm_accuracy_result(test_target,result,divnum)
        # accuracy=(tp+tn)/(tp+tn+fp+fn)
        # far=(fp)/(fp+tn)
        # frr=(fn)/(fn+tp)
        i=0.001
        far=1
        frr=0
        while far>frr:
            tp,tn,fp,fn=svm_accuracy_score(test_target,score,i,38)
            accuracy=(tp+tn)/(tp+tn+fp+fn)
            far=(fp)/(fp+tn)
            frr=(fn)/(fn+tp)
            # print("i=",i)
            # print("accuracy:",accuracy,"far:",far,"frr:",frr)
      
            i=i+0.001
        print("i=",i)
        print(tp,tn,fp,fn)
        accuracy=(tp+tn)/(tp+tn+fp+fn)
        far=(fp)/(fp+tn)
        frr=(fn)/(fn+tp)
        print("accuracy:",accuracy,"far:",far,"frr:",frr)

        meanacc.append(accuracy)
        meanfar.append(far)
        meanfrr.append(frr)
    print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
    for i in range(len(meanacc)):
        print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])





def keras_mlp(data,target,divnum):

    print(len(target))
    # print(target)
    input_shape=(17,1)
    print(input_shape)

    data=np.array(data)
    target=np.array(target)
    train_data,test_data, train_target, test_target = train_test_split(data,target,test_size = 0.2,random_state = 30,stratify=target)



    train_data,test_data,scale_mean,scale_scale=stdpro(train_data,test_data)
    train_data=train_data.reshape(train_data.shape[0], 17, 1)
    test_data=test_data.reshape(test_data.shape[0], 17, 1)
    print(train_data.shape,train_target.shape)
    print(test_data.shape,test_target.shape)

    # # 自定义层
    model=simplemoel_mlp(input_shape)

    model.summary()
    rms = RMSprop()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])


    history=model.fit(train_data, train_target,
              batch_size=128,epochs=40,
              validation_split=0.2)



    # compute final accuracy on training and test sets
    y_pred = model.predict(train_data)
    print(train_target)
    print(y_pred)
    y_pred = model.predict(test_data)
    print(test_target)
    print(y_pred)

    # tp,tn,fp,fn=mlp_accuracy_result(test_target,y_pred,divnum)
    # print(tp,tn,fp,fn)
    # accuracy=(tp+tn)/(tp+tn+fp+fn)
    # far=(fp)/(fp+tn)
    # frr=(fn)/(fn+tp)

    i=0.001
    far=1
    frr=0
    while far>frr:
        tp,tn,fp,fn=mlp_accuracy_score(test_target,y_pred,i,38)
        far=(fp)/(fp+tn)
        frr=(fn)/(fn+tp)
        # print("i=",i)
        # print("accuracy:",accuracy,"far:",far,"frr:",frr)
        i=i+0.001
    print("i=",i)
    print(tp,tn,fp,fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    far=(fp)/(fp+tn)
    frr=(fn)/(fn+tp)
    print("accuracy:",accuracy,"far:",far,"frr:",frr)





# data,target =loaddata()
# 无lda
# accuracy: 0.7882422859316313 far: 0.2114846553870944 frr: 0.22186088527551942
# accuracy: 0.7830171635049684 far: 0.21685099733880223 frr: 0.22186088527551942

# 有lda
# accuracy: 0.8629106641943612 far: 0.13686857589296614 frr: 0.14525745257452574
# accuracy: 0.8620786383302429 far: 0.13775726946458655 frr: 0.14399277326106594


data,target =loaddata_2()
# 无区分精度



sklearn_svmclass(data,target,38)
# keras_mlp(data,target,38)

