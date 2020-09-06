# -*- coding=utf-8 -*-
import numpy as np
from sklearn.decomposition import FastICA,PCA
from scipy.stats import kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from normal_tool import *
from sklearn.preprocessing import StandardScaler

#根据项目需要，自定义的工具


manhattan_distance = lambda x, y: np.abs(x - y)
euclidean_distance = lambda x, y: np.sqrt(np.power((x - y), 2))

#根据峰度判断干扰信号是那个
def maline(data1,data2):	
	if abs(kurtosis(data1))>abs(kurtosis(data2)):#高峰度的是ma信号
		ma=data1
		pulse=data2
	else:
		ma=data2	
		pulse=data1
	return ma,pulse


#将2个ppg数据源用fastica算法划分成MA段和PLUSE段
def ppgfica(data1,data2):
	S = np.c_[data1, data2]
	ica = FastICA(n_components=2)
	ica_dataset_X = ica.fit_transform(S)
	data1, data2=np.split(ica_dataset_X,[1],1)
	data1=data1.tolist()
	data1=[i[0] for i in data1]
	data2=data2.tolist()
	data2=[i[0] for i in data2]
	data1,data2=maline(data1,data2)
	return data1,data2

#根据分数选择分数最高的前topn个特征
def scoreselect(data,sort,topn=30):
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

#根据序列，计算信息熵，返回最高信息量的前i个特征列
def minepro(train_data,test_data,train_target,i):#训练集数据，测试集数据，训练集结果，降维后的特征数
	sort=minecal(train_data,train_target)
	train_data=scoreselect(train_data,sort,i)
	test_data=scoreselect(test_data,sort,i)
	return train_data,test_data,sort

#根据弹性网权重，使用权重最高的i个特征列
def elasticnetpro(train_data,test_data,train_target,i):#训练集数据，测试集数据，训练集结果，降维后的特征数
	sort=elasticnet(train_data,train_target)
	train_data=scoreselect(train_data,sort,i)
	test_data=scoreselect(test_data,sort,i)
	return train_data,test_data,sort

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


def pcapro(train_data,test_data):
	# pca = PCA(n_components=12,whiten=True)
	pca = PCA(n_components=12)
	pca = pca.fit(train_data)
	train_data=pca.transform(train_data)
	# print(np.dot(test_data[0]-lda.xbar_,lda.scalings_))
	#降维等于 np.dot(test_data-lda_bar,lda_scaling)
	if len(test_data)>0:
		test_data=pca.transform(test_data) 	
	# print(test_data[0])
	pca_mean=[i for i in pca.mean_]
	print("pca_mean:",pca_mean)
	pca_components=[[j for j in i] for i in pca.components_]
	print("pca_components:",pca_components)
	# pca_explained_variance_=np.sqrt(pca.explained_variance_)
	# print("pca_components:",pca_components)
	return train_data,test_data,pca_mean,pca_components

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


# 将array数据转成dic格式，给libsvm用,特征从1开始编码
def arraytodic(data):
	dic=[]
	length=len(data[0])
	for i in range(len(data)):
		temp={}
		for j in range(length):
			temp[(j+1)]=data[i][j]
		dic.append(temp)	
	return dic

# 数据扩展成2的幂次方，后面补0，给fft和小波变换使用
def to2power(data):
	length=len(data)
	flag=0
	if length<256:
		flag=256
	else:
		if length<512:
			flag=512
		else:
			flag=1024
	for i in range(flag-length):
		data.append(0)
	return data

#导数序列 
def interationcal(data):
	interation=[]
	for i in range(len(data)-1):
		# interation.append(round((data[i+1]-data[i]), 5))
		interation.append(data[i+1]-data[i])
	return interation

# 将array数据转成dic格式，给JS散度使用
def tagcal(data):
	tag={}
	for i in range(len(data)):
		tag[data[i]]=0
	return tag
# 复制dic格式，给JS散度使用
def tagsend(data):
	tag={}
	for i in data:
		tag[i]=0
	return tag

# 将一个序列变成分布，用于计算JS散度
def array_distribute_cal(data,tagdic):
	tag=tagsend(tagdic)
	for i in data:
		tag[i]=tag[i]+1
	score=[]
	for i in tag:
		if tag[i]==0:
			score.append(0.00000001)
		else:
			# score1.append(tag1[i])
			score.append(tag[i]/len(data))	
	return score

#手势识别方案中的短时能量计算方案
def short_time_energy(ppg):
	score=[]
	for i in range(0,len(ppg)-900):
		temp=np.mean(ppg[i:i+900])+3*np.std(ppg[i:i+900])
		score.append(temp)
	for i in range(len(ppg)-900,len(ppg)):
		temp=np.mean(ppg[i:])+3*np.std(ppg[i:])
		score.append(temp)
	return score

#手势识别方案中的能量计算方案
def energy(ppg):
	score=[]

	# for i in range(0,len(ppg)-240):
	# 	temp=np.mean(ppg[i:i+240])+3*np.std(ppg[i:i+900])
	# 	tempenergy=tempenergy+(data[j]-threshold)*data[j]
	# 	score.append(temp)
	
	threshold=0
	for i in range(0,len(ppg)-240):
		tempenergy=0
		for j in range(i,i+240):
			tempenergy=tempenergy+(ppg[j]-threshold)*ppg[j]
		score.append(tempenergy)	
	return score

#将行为传感器的数据扩充为原来的2倍
def sequence_incre(data):
	temp=[]
	for i in range(len(data)-1):
		temp.append(data[i])
		temp.append((data[i]+data[i+1])/2)
	return temp	

def sequence_to_300(data):
	data=sequence_incre(data)
	data=sequence_incre(data)
	datalen=len(data)

	inter=datalen/605
	temp=[]
	for i in range(605):
		temp.append(data[int(i*inter)])
	temp=meanfilt(temp,5)
	return temp

def filterparameterwrite(sort,lda_bar,lda_scaling,filename='./filterparameter.txt'):
	
	outputfile=open(filename,'w+')
	outputfile.write(str(sort))
	outputfile.write('\n')

	outputfile.write(str(lda_bar))
	outputfile.write('\n')
	
	outputfile.write(str(lda_scaling))

	outputfile.close()


def filterparameterread(filename='./filterparameter.txt'):
	
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