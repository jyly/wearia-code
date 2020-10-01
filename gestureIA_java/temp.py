
# -*- coding=utf-8 -*-
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis


def fine_grained_segment(dn,fre,threshold=1):
	oristd=[]
	winslen=200
	for i in range(len(dn)-winslen):
		ori=np.std(dn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)

	pointstartindex=0
	pointendindex=0
	tag=0
	i=len(oristd)-fre - 350
	lens=int(fre)
	while i >(lens+50):
		i=i-1
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>threshold:	
			flag=0
			print(i,oristd[i])
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(0,lens):
				if oristd[i+j]<threshold:
					flag=1
					break
			print(flag)
			if 0==flag:
				for j in range(0,lens):
					if oristd[i-j]>threshold+0.1:
						flag=1
						break
			print(flag)
			if 0==flag:
				pointstartindex=i-100
				pointendindex=i+200
				tag=0	
	return tag,pointstartindex,pointendindex

#根据峰度判断干扰信号是那个
def maline(data1,data2):	
	if abs(kurtosis(data1))>abs(kurtosis(data2)):#高峰度的是ma信号
		ma=data1
		pulse=data2
	else:
		ma=data2	
		pulse=data1
	return ma,pulse
def meanfilt(datalist,interval,adddis=1):
	filtlist=[]
	for i in range(0,len(datalist)-interval,adddis):
		filtlist.append(np.mean(datalist[i:i+interval]))
	# for i in range(0,interval):
	# 	filtlist.append(np.mean(datalist[i-interval:]))
	return filtlist

def highpass(high,fre,data,order=3):#只保留高于high频率的信号，fre是采样频率,order是滤波器阶数
	wh=high/(fre/2)
	b, a = signal.butter(order, wh, 'high')
	data = signal.filtfilt(b, a, data)
	return data

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

	
def bandpass(start,end,fre,data,order=3):#只保留start到end之间的频率的信号，fre是采样频率,order是滤波器阶数
	wa = start / (fre / 2) 
	we = end / (fre / 2) 
	b, a = signal.butter(order, [wa,we], 'bandpass')
	data = signal.filtfilt(b, a, data)
	return data

dirpath='./butter/'
filespace=os.listdir(dirpath)
for file in filespace:	
	filepath=dirpath+str(file)
	print(filepath)
	inputfile=open(filepath,'r+')
	feature=[]
	for i in inputfile:
		i=list(eval(i))
		feature.append(i)
	inputfile.close()	


	oristd=[]
	for i in range(len(feature[0])-200):
		ori=np.std(feature[0][i:i+200])
		oristd.append(ori)
	tag,pointstartindex,pointendindex=fine_grained_segment(feature[0],200,1)
	print(pointstartindex,pointendindex)
	# feature[0]=feature[0][:-1400]
	# feature[1]=feature[1][:-1400]
	plt.subplot(311)
	plt.plot(range(len(feature[0])), feature[0], 'red')
	plt.subplot(312)
	# plt.plot(range(len(feature[1])), feature[1], 'blue')
	plt.plot(range(len(oristd)), oristd, 'blue')
	plt.subplot(313)
	plt.plot(range(len(feature[0])), feature[0], 'red')
	plt.plot(range(len(feature[1])), feature[1], 'blue')
	plt.show()


# dirpath='./selected_oridata/clx_2/'
# filespace=os.listdir(dirpath)
# for file in filespace:	
# 	filepath=dirpath+str(file)
# 	print(filepath)
# 	inputfile=open(filepath,'r+')
# 	ppgx=[]
# 	ppgy=[]
# 	for i in inputfile:
# 		i=list(eval(i))
# 		if i[0]==2:
# 			ppgx.append(i[1])
# 			ppgy.append(i[2])
# 	inputfile.close()	

# 	# ppgx=ppgx[400:]
# 	# ppgy=ppgy[400:]
# 	ppgx=meanfilt(ppgx,20)
# 	orippgx=ppgx
# 	ppgx=highpass(2,200,ppgx)
# 	# ppgx=bandpass(1,3,200,ppgx)
# 	ppgy=meanfilt(ppgy,20)
# 	ppgy=highpass(2,200,ppgy)
# 	# ppgy=bandpass(1,3,200,ppgy)


# 	ppgx,ppgy=ppgfica(ppgx,ppgy)

# 	oristd=[]
# 	for i in range(len(ppgx)-200):
# 		ori=np.std(ppgx[i:i+200])
# 		oristd.append(ori)
# 	plt.subplot(311)
# 	plt.plot(range(len(ppgx)), ppgx, 'red')
# 	plt.subplot(312)
# 	plt.plot(range(len(orippgx)), orippgx, 'blue')
# 	plt.subplot(313)
# 	# plt.plot(range(len(ppgx)), ppgx, 'red')
# 	# plt.plot(range(len(ppgy)), ppgy, 'blue')
# 	plt.plot(range(len(oristd)), oristd, 'red')

# 	plt.show()
