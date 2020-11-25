
# -*- coding=utf-8 -*-
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis,skew
from sklearn import preprocessing
from scipy.signal import argrelmax,argrelmin

def minmaxscale(data):
	scaler = preprocessing.MinMaxScaler()
	data=scaler.fit_transform(np.array(data).reshape(-1,1))
	data=[i[0] for i in data]
	return data

def standardscale(data):
	scaler = preprocessing.StandardScaler()
	data=scaler.fit_transform(np.array(data).reshape(-1,1))
	data=[i[0] for i in data]
	return data

#获取峰值点
def find_extrema(signal):
    signal = np.array(signal)
    extrema_index = np.sort(np.unique(np.concatenate((argrelmax(signal)[0], argrelmin(signal)[0]))))
    extrema = signal[extrema_index]
    return zip(extrema_index.tolist(), extrema.tolist())


def fine_grained_segment(dn,fre,top,bottom):
	oristd=[]
	winslen=200
	tempdn=incretempdata(dn,int(fre/2))
	for i in range(len(tempdn)-winslen):
		ori=np.std(tempdn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)

	pointstartindex=0
	pointendindex=0
	tag=0

	i=len(oristd)-100
	lens=int(fre)
	while i >(lens):
		i=i-1
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>top:	
			flag=0
			finalcount=0
			#后面一定区间内的值大部份都小于阈值
			for j in range(0,100):
				if oristd[i+j]<top:
					finalcount=finalcount+1
			if finalcount<80:
				flag=1
			if 0==flag:
				#从后往前的一定区间内的值都大于阈值时，认为存在手势
				gesturecount=0
				for j in range(0,lens):
					if oristd[i-j]>top:
						gesturecount=gesturecount+1
				if gesturecount<150:
					flag=1
				

			if 0==flag:
				#前面的一段时间确认无手势的影响
				t=i-150
				# print(t)
				startinters=2*lens
				while t>startinters:
					t=t-1
					if oristd[t]<top:
						startcount=0
						for j in range(0,startinters):
							if oristd[t-j]<bottom:
								startcount=startcount+1
						# print(t,startcount)
						if startcount>350:
							tag=1
							pointendindex=i+20	
							pointstartindex=t-40
							break
		if tag==1:
			break		
	if (pointendindex-pointstartindex)>600:
		pointstartindex=0
		pointendindex=0
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


def incretempdata(data,incres):
	tempdata=[]
	for i in range(incres):
		tempdata.append(data[0])
	for i in data:
		tempdata.append(i)
	for i in range(incres):
		tempdata.append(data[-1])
	return tempdata

'''
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

	tempdata=incretempdata(feature[0],100)
	for i in range(len(tempdata)-200):
		ori=np.std(tempdata[i:i+200])
		oristd.append(ori)

	tag,pointstartindex,pointendindex=fine_grained_segment(feature[0],200,1.5,1)
	print(tag,pointstartindex,pointendindex)
	# tag,pointstartindex,pointendindex=fine_grained_segment(feature[0],200,1.2)
	# print(pointstartindex,pointendindex)
	# feature[0]=feature[0][:-1400]
	# feature[1]=feature[1][:-1400]
	plt.subplot(211)
	plt.plot(range(len(feature[0])), feature[0], 'red')
	plt.plot(range(len(feature[1])), feature[1], 'green')
	plt.subplot(212)
	# plt.plot(range(len(feature[1])), feature[1], 'blue')
	plt.plot(range(len(oristd)), oristd, 'blue')
	# plt.subplot(313)
	# feature[0]=highpass(2,200,feature[0])
	# feature[1]=highpass(2,200,feature[1])
	# plt.plot(range(len(feature[0])), feature[0], 'red')
	# plt.plot(range(len(feature[1])), feature[1], 'blue')
	plt.xlabel(filepath)
	# plt.title(filepath,fontsize=12,color='r')
	plt.show()

'''
dirpath='./oridata_2/tempuser4_1/'
filespace=os.listdir(dirpath)
for file in filespace:	
	filepath=dirpath+str(file)
	print(filepath)
	inputfile=open(filepath,'r+')
	starttag=0
	ppgx=[]
	ppgy=[]
	oldx=0
	oldy=0
	for i in inputfile:
		i=list(eval(i))
		if i[0]==2:
			if (0==i[1] or 0==i[2]):
				continue
			if(0==starttag):
				oldx=i[1]
				oldy=i[2]
				starttag=1
			x = abs(i[1]/oldx);
			y = abs(i[2]/oldy);
			# print(x)
			if(x<10 and x>0.1 and y<10 and y>0.1):
				ppgx.append(i[1])
				ppgy.append(i[2])
				oldx=i[1]
				oldy=i[2]
	inputfile.close()	
	if len(ppgx)<800:
		continue

	# ppgx=ppgx[:1000]
	# ppgy=ppgy[:1000]
	# ppgx=meanfilt(ppgx,20)
	# butterppgx=highpass(2,200,ppgx)


	# ppgx=minmaxscale(ppgx)
	# ppgy=minmaxscale(ppgy)
	# ppgx=standardscale(ppgx)
	# ppgy=standardscale(ppgy)


	butterppgx=bandpass(2,5,200,ppgx)
	butterppgy=bandpass(2,5,200,ppgy)
	butterppgx=minmaxscale(butterppgx)
	butterppgy=minmaxscale(butterppgy)
	# butterppgx=standardscale(butterppgx)
	# butterppgy=standardscale(butterppgy)
	



	icappgx,icappgy=ppgfica(butterppgx,butterppgy)
	# icappgx=minmaxscale(icappgx)
	# icappgy=minmaxscale(icappgy)


	tempx=incretempdata(icappgx,100)
	tempy=incretempdata(icappgy,100)

	oristd1=[]
	oristd2=[]
	orikur1=[]
	orikur2=[]
	oriskew1=[]
	oriskew2=[]
	for i in range(len(tempx)-200):
		ori=np.std(tempx[i:i+200])
		oristd1.append(ori)
		ori=np.std(tempy[i:i+200])
		oristd2.append(ori)	

		ori=abs(kurtosis(tempx[i:i+200]))
		orikur1.append(ori)
		ori=abs(kurtosis(tempy[i:i+200]))
		orikur2.append(ori)

		ori=abs(skew(tempx[i:i+200]))
		oriskew1.append(ori)
		ori=abs(skew(tempy[i:i+200]))
		oriskew2.append(ori)

	tag,pointstartindex,pointendindex=fine_grained_segment(icappgx,200,0.03,0.015)
	print(tag,pointstartindex,pointendindex)


	plt.subplot(4,2,1)
	plt.plot(range(len(ppgx)), ppgx, 'red')
	plt.subplot(4,2,2)
	plt.plot(range(len(ppgy)), ppgy, 'green')
	plt.subplot(4,2,3)
	plt.plot(range(len(butterppgx)), butterppgx, 'red')
	plt.subplot(4,2,4)
	plt.plot(range(len(butterppgy)), butterppgy, 'green')
	plt.subplot(4,2,5)
	plt.plot(range(len(icappgx)), icappgx, 'red')
	plt.subplot(4,2,6)
	plt.plot(range(len(icappgy)), icappgy, 'green')
	plt.subplot(4,2,7)
	for i in range(len(cD5)):
		plt.plot(i, cD5[i], '^')
	plt.plot(range(len(cD5)), cD5, 'red')
	plt.subplot(4,2,8)
	plt.plot(range(len(oristd2)), oristd2, 'green')
	plt.xlabel(str(file))
	plt.show()

	
	# butterppgx=bandpass(2,10,200,ppgx,3)
	# butterppgy=bandpass(2,10,200,ppgy,3)
	# icappgx,icappgy=ppgfica(butterppgx,butterppgy)

	# plt.subplot(611)
	# plt.plot(range(len(ppgx)), ppgx, 'red')
	# plt.plot(range(len(ppgy)), ppgy, 'green')
	# plt.subplot(612)

	# butterppgx=highpass(2,200,ppgx)
	# butterppgy=highpass(2,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# for i in range(len(pointindex)):
	# 	plt.plot(pointindex[i],butterppgx[pointindex[i]],'o')
	# plt.subplot(613)
	# butterppgx=[i for i in butterppgx]
	# print(butterppgx)

	# butterppgy=[i for i in butterppgy]
	# print(butterppgy)
	# butterppgx=meanfilt(butterppgx,20)
	# butterppgy=meanfilt(butterppgy,20)
	# icappgx=meanfilt(icappgx,20)

	# plt.plot(range(len(icappgx)), icappgx, 'red')
	# plt.plot(range(len(icappgy)), icappgy, 'green')
	# plt.subplot(614)
	# icappgx,icappgy=ppgfica(icappgx,icappgy)
	# plt.plot(range(len(icappgx)), icappgx, 'red')
	# plt.plot(range(len(icappgy)), icappgy, 'green')
	# plt.plot(range(len(oristd1)), oristd1, 'red')
	# plt.plot(range(len(oristd2)), oristd2, 'green')
	# plt.subplot(615)
	# icappgx,icappgy=ppgfica(icappgx,icappgy)
	# plt.plot(range(len(icappgx)), icappgx, 'red')
	# plt.plot(range(len(icappgy)), icappgy, 'green')
	# plt.plot(range(len(orikur1)), orikur1, 'red')
	# plt.plot(range(len(orikur2)), orikur2, 'green')
	# plt.subplot(616)
	# icappgx,icappgy=ppgfica(icappgx,icappgy)
	# plt.plot(range(len(icappgx)), icappgx, 'red')
	# plt.plot(range(len(icappgy)), icappgy, 'green')
	# plt.plot(range(len(oriskew1)), oriskew1, 'red')
	# plt.plot(range(len(oriskew2)), oriskew2, 'green')
	# plt.xlabel(str(file))
	# plt.show()

	# plt.subplot(611)
	# plt.plot(range(len(ppgx)), ppgx, 'red')
	# plt.plot(range(len(ppgy)), ppgy, 'green')
	# plt.subplot(612)
	# butterppgx=highpass(2,200,ppgx)
	# butterppgy=highpass(2,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # plt.plot(range(len(icappgx)), icappgx, 'red')
	# # plt.plot(range(len(icappgy)), icappgy, 'green')
	# # plt.xlabel("highpass 2")
	# plt.subplot(613)
	# butterppgx=bandpass(2,10,200,ppgx)
	# butterppgy=bandpass(2,10,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # plt.xlabel("bandpass 2,10")
	# plt.subplot(614)
	# butterppgx=bandpass(0.5,10,200,ppgx)
	# butterppgy=bandpass(0.5,10,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # plt.xlabel("bandpass 0.5,10")
	# plt.subplot(615)
	# butterppgx=bandpass(0.5,2,200,ppgx)
	# butterppgy=bandpass(0.5,2,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # plt.xlabel("bandpass 0.5,2")
	# plt.subplot(616)
	# butterppgx=bandpass(2,4,200,ppgx)
	# butterppgy=bandpass(2,4,200,ppgy)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # plt.xlabel("bandpass 5,10")
	# plt.xlabel(str(file))
	# plt.show()



	# pointindex=[]
	# last_single_waveform_start_index=None
	# last_extremum_index = None
	# last_extremum = None
	# threshold = (max(butterppgx) - min(butterppgx)) * 0.5
	# for extremum_index, extremum in find_extrema(butterppgx):
	# 	print(extremum_index)
	# 	if last_extremum is not None and extremum - last_extremum > threshold:
	# 		if last_single_waveform_start_index is not None:
	# 			pointindex.append(last_single_waveform_start_index)
	# 		last_single_waveform_start_index = last_extremum_index
	# 	last_extremum_index = extremum_index
	# 	last_extremum = extremum
	# import pywt
	# coef, freqs=pywt.cwt(ppgx,np.arange(1,32),wavelet='mexh')
	# print(len(coef[0]))
	# print(freqs)
	# import pywt
	# coeffs = pywt.wavedec(butterppgx,'haar',level=6)
	# cA6, cD6,cD5,cD4,cD3, cD2 , cD1=coeffs
