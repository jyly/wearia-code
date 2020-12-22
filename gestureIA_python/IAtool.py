# -*- coding=utf-8 -*-
import numpy as np
from sklearn.decomposition import FastICA,PCA
from scipy.stats import kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from normal_tool import *
from sklearn.preprocessing import StandardScaler
from dtw import dtw
from scipy.signal import argrelmax
import math
import pandas as pd
from pymrmre import mrmr



#根据项目需要，自定义的工具
manhattan_distance = lambda x, y: np.abs(x - y)
euclidean_distance = lambda x, y: np.sqrt(np.power((x - y), 2))

#获取峰值点
def find_extrema(signal):
    signal = np.array(signal)
    extrema_index = np.sort(np.unique(np.concatenate((argrelmax(signal)[0], argrelmin(signal)[0]))))
    extrema = signal[extrema_index]
    return zip(extrema_index.tolist(), extrema.tolist())

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

#根据序列，计算信息熵，返回最高信息量的前i个特征列
def minepro(train_data,test_data,train_target,i):#训练集数据，测试集数据，训练集结果，降维后的特征数
	sort=minecal(train_data,train_target)
	train_data=scoreselect(train_data,sort,i)
	test_data=scoreselect(test_data,sort,i)
	return train_data,test_data,sort

def mrmrpro(train_data,testdata,train_target,i):
	columns=[str(i) for i in range(len(train_data[0]))]
	df_train = pd.DataFrame(train_data,columns=columns)
	df_test = pd.DataFrame(testdata,columns=columns)
	df_target = pd.DataFrame(train_target,columns=['class'])
	solutions = mrmr.mrmr_ensemble(features=df_train,targets=df_target,solution_length=i)
	df_train=df_train[solutions[0][0]]
	df_test=df_test[solutions[0][0]]
	sort=[int(i) for i in solutions[0][0]]
	print("mrmrsort:",sort)

	df_train=df_train.values.tolist()
	df_test=df_test.values.tolist()
	return df_train,df_test,sort


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
	pca = PCA(n_components=20)
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
	tempdata=[]
	for i in data:
		tempdata.append(i)
	flag=0
	if length<256:
		flag=256
	else:
		if length<512:
			flag=512
		else:
			flag=1024
	for i in range(flag-length):
		tempdata.append(0)
	return tempdata

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
	temp.append(data[-1])	
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

#把数据重采样为指定长度
def data_resize(data,resize):
	datalens=len(data[0])
	inters=float(datalens)/resize
	temp=[[] for i in range(len(data))]
	for i in range(resize):
		for j in range(len(data)):
			temp[j].append(data[j][int(i*inters)])
	return temp


def datainner(data):
	innersize=len(str(round(data[0])))
	data=np.array(data)
	temp=data/100000
	# temp=data/math.pow( 10, (innersize-1))
	return temp


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


def mulfilterparameterwrite(sort1,sort2,lda_bar,lda_scaling,filename):
	
	outputfile=open(filename,'w+')
	outputfile.write(str(sort1))
	outputfile.write('\n')
	outputfile.write(str(sort2))
	outputfile.write('\n')
	outputfile.write(str(lda_bar))
	outputfile.write('\n')
	outputfile.write(str(lda_scaling))
	outputfile.close()


def mulfilterparameterread(filename):
	
	inputfile=open(filename,'r+')
	parameter=[]
	for i in inputfile:
		i=list(eval(i))
		parameter.append(i)
	inputfile.close()	
	sort1=parameter[0]
	sort2=parameter[1]
	lda_bar=parameter[2]
	lda_scaling=parameter[3]
	return sort1,sort2,lda_bar,lda_scaling

#获取单个数据列中由小到大排序的距离数值
def calgestureprofile(data):
	# print(len(data))
	lens=len(data)
	dist=[[[] for i in range(lens)] for j in range(lens)]
	for i in range(lens):
		dist[i][i]=0
	for i in range(lens-1):
		for j in range(i+1,lens):
			d, cost_matrix, acc_cost_matrix, path = dtw(data[i], data[j], dist=manhattan_distance)
			print("i=",i,"j=",j,"d=",d)
			dist[i][j]=d
			dist[j][i]=d
	meandist=[np.mean(dist[i]) for i in range(lens)]
	print(meandist)
	distsort = np.argsort(meandist)#由小到大排序，得到对应的序号
	temp=[i for i in distsort]
	distsort=temp
	print("distsort：",distsort)
	samples=[]
	samples.append(data[distsort[0]])
	samples.append(data[distsort[1]])
	samples.append(data[distsort[2]])
	return samples

def caldtw(a,b):
	d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=manhattan_distance)
	return d



#将队列数据化为目标分类的队列
def listtodic(data,target):
	index=0
	dicdata=[[]]
	for i in range(len(target)):
		if target[i]==(index+1):
			dicdata[index].append(data[i])
		else:
			index=index+1
			dicdata.append([])
	return dicdata

#将目标分类的队列化为队列数据
def dictolist(data):
	feature=[]
	target=[]
	index=1
	for i in range(len(data)):
		for j in range(len(data[i])):
			feature.append(data[i][j])
			target.append(index)
		index=index+1	
	index=index-1	
	return feature,target,index







def same_gesture_selected(feature,target,targetnum,selectnumber=1):# 数据，标记，选择的手势

	dicdata=listtodic(feature,target)
	#选择对应的手势
	selectfeature=[]
	selectedtarget=[]
	#选择的手势
	index=1
	while selectnumber<targetnum:
		for i in dicdata[selectnumber]:
			selectfeature.append(i)
			selectedtarget.append(index)
		selectnumber=selectnumber+9
		index=index+1
	
	# print(selectfeature[0])
	# print(selectedtarget)
	# print(len(selectedtarget))
	feature=selectfeature
	feature=np.array(feature)
	target=selectedtarget
	targetnum=index-1
	return feature,target,targetnum


def datashape(train_data,test_data,train_target,test_target):
	train_data=np.array(train_data)
	train_target=np.array(train_target)
	test_data=np.array(test_data)
	test_target=np.array(test_target)
	print("train_data.shape:",train_data.shape)
	print("test_data.shape:",test_data.shape)
	return train_data,test_data,train_target,test_target


def datacombine(ppg_data,motion_data):
	data=[]
	for i in range(len(ppg_data)):
		temp=[]
		for j in ppg_data[i]:
			temp.append(j)
		for j in motion_data[i]:
			temp.append(j)
		data.append(temp)
	return data

#数据转置，将n*m转为m*n
def datatranspose(data):
	data=np.array(data)
	tempdata=[]
	for i in range(len(data)):
		tempdata.append(data[i].T)
	return tempdata	

#对样本对进行额外处理
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



def datatopic(data):
	tempdata=[]
	for i in range(len(data)):
		temp=[]
		for j in range(len(data[i])):
			pic=recurrenceplot(data[i][j])
			temp.append(pic)
		tempdata.append(temp)
	return tempdata

def pairtopic(data):
	tempdata=[]
	for i in range(len(data)):
		pair=[]
		for j in range(len(data[i])):
			# pic=recurrenceplot(data[i][j])
			pic=gramianplot(data[i][j])
			pic=np.array(pic)
			# print(pic.shape)
			pic=pic.transpose(1,2,0)
			pair.append(pic)
		pair=np.array(pair)
		# print(pair.shape)

		tempdata.append(pair)
	return tempdata




