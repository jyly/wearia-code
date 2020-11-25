
# -*- coding=utf-8 -*-
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis,skew

import heartpy as hp

from scipy.signal import argrelmax,argrelmin



#获取峰值点
def find_extrema(signal):
    signal = np.array(signal)
    extrema_index = np.sort(np.unique(np.concatenate((argrelmax(signal)[0], argrelmin(signal)[0]))))
    extrema = signal[extrema_index]
    return zip(extrema_index.tolist(), extrema.tolist())


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

	i=len(oristd)-fre - 50
	lens=int(fre)
	while i >(lens+50):
		i=i-1
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>threshold:	
			flag=0
			# print(i,oristd[i])
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(0,lens):
				if oristd[i+j]<threshold:
					flag=1
					break
			# print(flag)
			if 0==flag:
				#前面的一段时间确认无手势的影响
				for j in range(0,lens):
					if oristd[i-j]>threshold+0.1:
						flag=1
						break
			# print(flag)
			if 0==flag:
				pointstartindex=i-50
				for j in range(i+lens,len(oristd)-lens):
					flag=0
					for k in range(j,j+lens):
						if oristd[k]>threshold+0.1:
							flag=1
							break
					if 0==flag:
						pointendindex=j+50
						tag=1
						break
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


def pointcal(data):

	k_slope=[]
	for i in range(len(data)-1):
		k_slope.append(data[i+1]-data[i])

	
	maxp=[]
	minn=[]

	start=np.argmax(data[0:100])

	maxp.append(start)
	#1时寻找局部最低点，0时寻找局部最高点
	maxin=1

	while i<len(data)-50:
		#寻找最低点
		if 1==maxin and k_slope[i]>0:
			count=0
			for j in range(i+1,i+40):
				if k_slope[j]>0:
					count=count+1
			if count>20:
				stop=np.argmin(data[i-20:i+20])+i-20
				minn.append(stop)
				i=stop+20
				maxin=0
		
		#寻找最高点
		if 0==maxin and k_slope[i]<0:
			count=0
			for j in range(i+1,i+40):
				if k_slope[j]<0:
					count=count+1
			if count>20:
				stop=np.argmax(data[i-20:i+20])+i-20	
				maxp.append(stop)
				i=stop+20
				maxin=1
		i=i+1

	return maxp,minn



dirpath='./selected_madata/'
filespace=os.listdir(dirpath)
for file in filespace:	
	filepath=dirpath+str(file)
	print(filepath)
	inputfile=open(filepath,'r+')
	data=[]
	temp=[]
	for i in inputfile:
		i=list(eval(i))
		if len(temp)<2:
			temp.append(i)
		else:
			data.append(temp)
			temp=[]
			temp.append(i)	
	inputfile.close()	

	for i in range(len(data)):
		ppgx=data[i][0]
		ppgy=data[i][1]

		plt.subplot(611)
		plt.plot(range(len(ppgx)), ppgx, 'red')
		plt.plot(range(len(ppgy)), ppgy, 'green')
		plt.subplot(612)
		butterppgx=highpass(2,200,ppgx)
		butterppgy=highpass(2,200,ppgy)
		plt.plot(range(len(butterppgx)), butterppgx, 'red')
		plt.plot(range(len(butterppgy)), butterppgy, 'green')
		# plt.plot(range(len(icappgx)), icappgx, 'red')
		# plt.plot(range(len(icappgy)), icappgy, 'green')
		# plt.xlabel("highpass 2")
		plt.subplot(613)
		butterppgx=bandpass(2,10,200,ppgx)
		butterppgy=bandpass(2,10,200,ppgy)
		plt.plot(range(len(butterppgx)), butterppgx, 'red')
		plt.plot(range(len(butterppgy)), butterppgy, 'green')
		# plt.xlabel("bandpass 2,10")
		plt.subplot(614)
		butterppgx=bandpass(2,4,200,ppgx)
		butterppgy=bandpass(2,4,200,ppgy)
		plt.plot(range(len(butterppgx)), butterppgx, 'red')
		plt.plot(range(len(butterppgy)), butterppgy, 'green')
		# plt.xlabel("bandpass 0.5,10")
		plt.subplot(615)


		maxp,minn=pointcal(butterppgx)
		for j in range(len(maxp)):
			plt.plot(maxp[j],butterppgx[maxp[j]],'o')
		for j in range(len(minn)):
			plt.plot(minn[j],butterppgx[minn[j]],'^')

		# sp=[]
		# dn=[]	
		# dp=[]
		# ef=[]
		# sf=[]
		# butterppgx=bandpass(2,10,200,ppgx)
		# start=np.argmax(butterppgx[0:200])
		# startindex=0
		# for j in range(len(maxp)):
		# 	if abs(maxp[j]-start)<20:
		# 		startindex=j
		# 		break
		# print(startindex)		
		# while startindex<(len(maxp)-2):
		# 	sp.append(maxp[startindex])
		# 	dn.append(minn[startindex])
		# 	dp.append(maxp[startindex+1])
		# 	ef.append(minn[startindex+1])
		# 	sf.append(minn[startindex+1])
		# 	startindex=startindex+2
		# print(sp)		
		# print(dp)		
		# print(dn)		
		# print(ef)		

		# for j in range(len(sp)):
		# 	plt.plot(sp[j],butterppgx[sp[j]],'o','red')
		# for j in range(len(dp)):
		# 	plt.plot(dp[j],butterppgx[dp[j]],'o',color='green')

		# for j in range(len(dn)):
		# 	plt.plot(dn[j],butterppgx[dn[j]],'^',color='red')
		# for j in range(len(ef)):
		# 	plt.plot(ef[j],butterppgx[ef[j]],'^','green')
		# maxp,minn=pointcal(butterppgy)
		# for j in range(len(maxp)):
		# 	plt.plot(maxp[j],butterppgy[maxp[j]],'o')	
		# for j in range(len(minn)):
		# 	plt.plot(minn[j],butterppgy[minn[j]],'^')


		plt.plot(range(len(butterppgx)), butterppgx, 'red')
		plt.plot(range(len(butterppgy)), butterppgy, 'green')

		plt.subplot(616)
		butterppgx=bandpass(2,8,200,ppgx)
		butterppgy=bandpass(2,8,200,ppgy)
		butterppgx,butterppgy=ppgfica(butterppgx,butterppgy)
		plt.plot(range(len(butterppgx)), butterppgx, 'red')
		plt.plot(range(len(butterppgy)), butterppgy, 'green')
		# plt.xlabel("bandpass 5,10")
		plt.xlabel(str(file))
		plt.show()
	# print(len(ppgx))
	# ppgx=ppgx[:600]
	# ppgy=ppgy[:600]
	# ppgx=ppgx[1200:1500]
	# ppgy=ppgy[1200:1500]
	# ppgx=meanfilt(ppgx,20)
	# butterppgx=highpass(2,200,ppgx)
	# butterppgx=bandpass(2,10,200,ppgx,3)
	# butterppgx=bandpass(5,8,200,ppgx,3)
	# ppgy=meanfilt(ppgy,20)
	# butterppgy=highpass(2,200,ppgy)
	# butterppgy=bandpass(2,10,200,ppgy,3)
	# butterppgy=bandpass(5,8,200,ppgy,3)

	# icappgx,icappgy=ppgfica(butterppgx,butterppgy)
	# ppgx,ppgy=ppgfica(ppgx,ppgy)


	# icappgx=bandpass(2,10,200,ppgx,3)
	# icappgy=bandpass(2,10,200,ppgy,3)


	# oristd1=[]
	# oristd2=[]
	# orikur1=[]
	# orikur2=[]
	# oriskew1=[]
	# oriskew2=[]
	# for i in range(len(butterppgx)-200):
	# 	ori=np.std(butterppgx[i:i+200])
	# 	oristd1.append(ori)
	# 	ori=np.std(butterppgy[i:i+200])
	# 	oristd2.append(ori)	

	# 	ori=abs(kurtosis(butterppgx[i:i+200]))
	# 	orikur1.append(ori)
	# 	ori=abs(kurtosis(butterppgy[i:i+200]))
	# 	orikur2.append(ori)

	# 	ori=abs(skew(butterppgx[i:i+200]))
	# 	oriskew1.append(ori)
	# 	ori=abs(skew(butterppgy[i:i+200]))
	# 	oriskew2.append(ori)

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


	# plt.subplot(611)
	# plt.plot(range(len(ppgx)), ppgx, 'red')
	# plt.plot(range(len(ppgy)), ppgy, 'green')
	# plt.subplot(612)
	# plt.plot(range(len(butterppgx)), butterppgx, 'red')
	# plt.plot(range(len(butterppgy)), butterppgy, 'green')
	# # for i in range(len(pointindex)):
	# # 	plt.plot(pointindex[i],butterppgx[pointindex[i]],'o')

	# plt.subplot(613)
	# plt.plot(range(len(icappgx)), icappgx, 'red')
	# plt.plot(range(len(icappgy)), icappgy, 'green')
	# plt.subplot(614)
	# plt.plot(range(len(oristd1)), oristd1, 'red')
	# plt.plot(range(len(oristd2)), oristd2, 'green')
	# plt.subplot(615)
	# plt.plot(range(len(orikur1)), orikur1, 'red')
	# plt.plot(range(len(orikur2)), orikur2, 'green')
	# plt.subplot(616)
	# plt.plot(range(len(oriskew1)), oriskew1, 'red')
	# plt.plot(range(len(oriskew2)), oriskew2, 'green')
	# plt.xlabel(str(file))
	# plt.show()

