# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU

import random

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from keras.utils.np_utils import to_categorical

#计算多分类的准确度
def mul_accuracy_result(target,result,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0

    for i in range(0,divnum):
        for j in range(len(result)):
            if np.argmax(result[j])==i:
                if target[j][0]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j][0]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn


def simplemoel_mlp(input_shape,divnum):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=(input_shape), name='input')
    
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    
    x = Dense(41,activation='sigmoid', name='output')(x)

    outputs=x

    return Model(inputs, outputs)


def conv_lstm(input_shape,divnum):
    inputs = Input(shape=(input_shape), name='input')

    # x = Reshape((2,200,1))(inputs)
    # x = Conv2D(4, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x) 
    # x = Conv2D(1, (1, 3), activation='relu',padding='same')(x)
    # x = Dropout(0.1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x) 

    # x = Reshape((200,1))(inputs)
    # x = Reshape((200,1))(inputs)
    x = Permute((2,1))(inputs)
    x = CuDNNLSTM(units=32,input_shape=(200,2),return_sequences=True)(x)
    x = CuDNNLSTM(units=32,return_sequences=True)(x)
    x = CuDNNLSTM(units=32)(x)
    # x = CuDNNLSTM(units=32,input_shape=(200,2))(x)
    # x = CuDNNLSTM(units=8,input_shape=(100,2),return_sequences=True)(x)
    # x = Flatten()(x)

    x = Dense(41,activation='sigmoid',name='output')(x)

    outputs=x
    return Model(inputs, x)

# def simplemoel_mlp(input_shape,divnum):
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     inputs = Input(shape=(input_shape), name='input')
    
#     x = Flatten()(inputs)
#     x = Dense(128)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(128)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(128)(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     outputs=x
#     x = Dense(41,activation='sigmoid')(x)

#     return Model(inputs, x)

def keras_mulclass(featureset,target,divnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)
		# train_data,test_data, train_target, test_target = train_test_split(featureset,target,train_size = 162,random_state = t*30,stratify=target)

		train_data=np.array(train_data)
		test_data=np.array(test_data)
		train_target=np.array(train_target)
		test_target=np.array(test_target)
		# train_target=train_target.reshape(-1,1)
		# test_target=test_target.reshape(-1,1)
		print(train_target[0])
		train_target = to_categorical(train_target)
		test_target = to_categorical(test_target)
		print(train_target[0])

		print(train_data.shape)
		print(test_data.shape)
		print(train_target.shape)
		print(test_target.shape)
		print("进入第",t,"轮分类阶段")
		input_shape=(2,200)
		# model=simplemoel_mlp(input_shape,divnum)
		model=conv_lstm(input_shape,divnum)

		model.summary()
		rms = RMSprop()
		# model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
		model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
		history=model.fit(train_data, train_target,
		          batch_size=2048,epochs=30,
		          validation_split=0.2)
		result = model.predict(test_data)

		print('原结果：',test_target)
		print('预测结果：',result[0])

	

		tp,tn,fp,fn=mul_accuracy_result(test_target,result,divnum)
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