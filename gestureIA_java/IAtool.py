# -*- coding=utf-8 -*-
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import math
from scipy import stats
from minepy import MINE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tftb.processing import WignerVilleDistribution,PseudoWignerVilleDistribution
from sklearn_porter import Porter

#计算相关系数
def calc_corr(a, b):
	a_avg = sum(a)/len(a)	
	b_avg = sum(b)/len(b) 	
	# 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n	
	cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)]) 	
	# 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
	sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b])) 	
	corr_factor = cov_ab/sq 	
	return corr_factor

#均值滤波
def meanfilt(datalist,interval,adddis=1):
	filtlist=[]
	for i in range(0,len(datalist)-interval,adddis):
		filtlist.append(np.mean(datalist[i:i+interval]))
	return filtlist

def highpass(high,fre,order=3):#只保留高于high频率的信号，fre是采样频率,order是滤波器阶数
	wh=high/(fre/2)
	b, a = signal.butter(order, wh, 'high')
	return b,a

def lowpass(low,fre,order=3):#只保留低于low频率的信号，fre是采样频率,order是滤波器阶数
	wl=high/(fre/2)
	b, a = signal.butter(order, wl, 'low')
	return b,a

def bandpass(start,end,fre,order=3):#只保留start到end之间的频率的信号，fre是采样频率,order是滤波器阶数
	wa = start / (fre / 2) 
	we = end / (fre / 2) 
	b, a = signal.butter(order, [wa,we], 'bandpass')
	return b,a

def indexpicshow(data):
	plt.plot(range(len(data)), data, 'blue')
	plt.show()

def mixindexpicshow(data1,data2):
	plt.subplot(311)
	plt.plot(range(len(data1)), data1, 'red')
	plt.subplot(312)
	plt.plot(range(len(data2)), data2, 'blue')
	plt.subplot(313)
	plt.plot(range(len(data1)), data1, 'red')
	plt.plot(range(len(data2)), data2, 'blue')
	plt.show()

def indexpicsave(data,savefile):
	fig = plt.figure()
	plt.plot(range(len(data)), data, 'blue')
	fig.savefig(savefile)
	plt.close()

def mixindexpicsave(data1,data2,savefile):
	fig = plt.figure()
	plt.subplot(311)
	plt.plot(range(len(data1)), data1, 'red')
	# plt.axhline(20000000)
	plt.subplot(312)
	plt.plot(range(len(data2)), data2, 'blue')
	# plt.axhline(20000000)
	plt.subplot(313)
	plt.plot(range(len(data1)), data1, 'red')
	plt.plot(range(len(data2)), data2, 'blue')
	# plt.axhline(20000000)
	fig.savefig(savefile)
	plt.close()



def finalmixindexpicsave(data1,data2,start,end,savefile):
	fig = plt.figure()
	plt.subplot(311)
	plt.plot(range(len(data1)), data1, 'red')
	plt.axvline(start)
	plt.subplot(312)
	plt.plot(range(len(data2)), data2, 'blue')
	plt.axvline(start)
	plt.subplot(313)
	plt.plot(range(len(data1)), data1, 'red')
	plt.plot(range(len(data2)), data2, 'blue')
	plt.axvline(start)
	fig.savefig(savefile)
	plt.close()


def picshow(datax,datay):
	plt.plot(datax, datay, 'blue')
	plt.show()	



def insertfilt(data,timestamp,fre):
	# picshow(timestamp,data)
	sample_interval=1000/fre
	filtdata=[]
	lasttime=timestamp[0]
	lastdata=data[0]
	localtime=timestamp[0]+sample_interval
	for i in range(len(timestamp)):
		if localtime==timestamp[i]:
			lastdata=data[i]
			lasttime=localtime
			localtime=localtime+sample_interval
			filtdata.append(data[i])
		else:
			if localtime<timestamp[i]:
				# print (localtime,timestamp[i],lasttime)
				k=(data[i]-lastdata)/(timestamp[i]-lasttime)
				lastdata=lastdata+k*sample_interval
				filtdata.append(lastdata)
				lasttime=localtime
				localtime=localtime+sample_interval
	return filtdata







#ica处理
def fica(data1,data2):
	S = np.c_[data1, data2]
	ica = FastICA(n_components=2)
	ica_dataset_X = ica.fit_transform(S)
	data1, data2=np.split(ica_dataset_X,[1],1)
	data1=data1.tolist()
	data1=[i[0] for i in data1]
	data2=data2.tolist()
	data2=[i[0] for i in data2]
	return data1,data2

#根据峰度判断干扰信号是那个
def maline(data1,data2):	
	if abs(stats.kurtosis(data1))>abs(stats.kurtosis(data2)):#高峰度的是dn，ma信号
		dn=data1
		xn=data2
	else:
		dn=data2	
		xn=data1
	return dn,xn

#快傅里叶变换，获取频域和对应的振幅
def fft(data, sampling_rate, fft_size=None):  
    if fft_size is None:  
        fft_size = len(data)  
    data = data[:fft_size]  
    datafft = abs(np.fft.rfft(data)/fft_size)  
    freqs = np.linspace(0, int(1.0*sampling_rate/2), int(1.0*fft_size/2+1))    #linspace(0,100,501)
    return freqs, datafft    #频域，对应振幅

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
	informsort = np.argsort(informscore)#由小到大排序
	return informsort

#根据信息熵选择特征
def mineselect(data,informsort,topn=30):
	if topn>len(data[0]):
		topn=len(data[0])
	selectdata=[]
	for i in range(len(data)):
		selectdata.append([])
		for j in range(topn):
			selectdata[i].append(data[i][informsort[-1-j]])
	return selectdata

#根据序列，计算信息熵，返回最高信息量的前i个特征列
def minepro(train_data,test_data,train_target,i):#训练集数据，测试集数据，训练集结果，降维后的特征数
	informsort=minecal(train_data,train_target)
	train_data=mineselect(train_data,informsort,i)
	test_data=mineselect(test_data,informsort,i)
	return train_data,test_data


def ldapro(train_data,test_data,train_target):
	lda = LinearDiscriminantAnalysis()
	lda = lda.fit(train_data, train_target)
	train_data=lda.transform(train_data)
	test_data=lda.transform(test_data) 	
	# modelsave(lda,"./lda.model")

	return train_data,test_data

def pcapro(train_data,test_data,i=10):
	pca = PCA(n_components=i)
	pca = pca.fit(train_data)
	train_data=pca.transform(train_data)
	test_data=pca.transform(test_data) 
	return train_data,test_data


def WignerVillecal(data):	
	dist = PseudoWignerVilleDistribution(data)
	result = dist.run()
	tfr, times, freqs=result
	x=[]
	y=[]
	z=[]
	# print(len(times))
	# print(len(freqs))
	# print(len(tfr),len(tfr[0]))
	for i in range(len(times)):
		for j in range(len(freqs)):
			# if tfr[j][i]>0.05:
			x.append(times[i])#时域序号点
			y.append(freqs[j])#频域序号点
			z.append(tfr[j][i])
	return x,y,z


def arraytodic(data):
	dic=[]
	for i in range(len(data)):
		temp={}
		for j in range(len(data[0])):
			temp[j]=data[i][j]
		dic.append(temp)	
	return dic

def modelsave(model,modelname):

	porter = Porter(model, language='java')
	output = porter.export(embed_data=True)
	print(output)

	f = open(modelname,'w+')
	for line in output.chunks():
		f.write(line)
	f.close()