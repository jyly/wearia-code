
# -*- coding=utf-8 -*-
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis,skew
from sklearn import preprocessing
from normal_tool import *


def JS_incretempdata(data,incres):
	tempdata=[]
	for i in range(incres):
		tempdata.append(data[i])
	for i in data:
		tempdata.append(i)
	for i in range(incres):
		tempdata.append(data[-1-i])
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
			# score.append(tag[i])
			score.append(tag[i]/len(data))	
	return score


def coarse_grained_detect(ppg,threshold=1):
	# indexpicshow(ppg)
	# ppg=minmaxscale(ppg)

	ppg=meanfilt(ppg,10)

	ppginter=interationcal(ppg)
	# print(ppginter)
	# ppginter=[round(i,6) for i in ppg]
	# ppginter=meanfilt(ppginter,20)

	ppginter=minmaxscale(ppginter)
	# ppginter=standardscale(ppginter)
	# ppginter=minmaxscale(ppg)
	# ppginter=standardscale(ppg)
	orippginter=[round(i,1) for i in ppginter]

	# indexpicshow(ppginter)
	# print(ppginter)

	# score=IAtool.energy(ppginter)
	# indexpicshow(score)

	alltag=tagcal(orippginter)
	ppginter=JS_incretempdata(orippginter,200)
	ppginter=meanfilt(ppginter,10)
	JS=[]
	for i in range(0,len(ppginter)-400):
		tempinter=standardscale(ppginter[i:i+400])
		alltag=tagcal(tempinter)

		score1=array_distribute_cal(tempinter[0:200],alltag)
		score2=array_distribute_cal(tempinter[200:400],alltag)

		# score1=array_distribute_cal(ppginter[i:i+200],alltag)
		# score2=array_distribute_cal(ppginter[i+200:i+400],alltag)	
		tempjs=JS_divergence(score1,score2)
		# tempjs=cos_distance(score1,score2)
		if(i==30):
			print(score1)
			print(score2)
		# tempjs=calc_corr(score1,score2)
		# tempjs=jaccard_distance(ppginter[i:i+200],ppginter[i+200:i+400])
		JS.append(tempjs)
	# print(JS)	
	# ppginter=meanfilt(ppginter,40)
	# indexpicshow(ppg)

	# pointpicshow(JS)
	# mixindexpicshow(JS,orippginter)

	return JS

def incretempdata(data,incres):
	tempdata=[]
	for i in range(incres):
		tempdata.append(data[0])
	for i in data:
		tempdata.append(i)
	for i in range(incres):
		tempdata.append(data[-1])
	return tempdata
	
def calenergy(data):
	tempdata=incretempdata(data,100)
	energy=[]
	for i in range(len(data)):
		energy.append(np.mean(tempdata[i:i+200])+3*np.std(tempdata[i:i+200]))
		# energy.append(np.std(tempdata[i:i+200]))
	return energy

def bandpass(start,end,fre,data,order=3):#只保留start到end之间的频率的信号，fre是采样频率,order是滤波器阶数
	wa = start / (fre / 2) 
	we = end / (fre / 2) 
	b, a = signal.butter(order, [wa,we], 'bandpass')
	data = signal.filtfilt(b, a, data)
	return data


# dirpath='./selected_madata/'
# filespace=os.listdir(dirpath)
# for file in filespace:	

# if 1==1:	
# 	# filepath=dirpath+str(file)
# 	filepath='./selected_madata/temp.csv'
# 	print(filepath)
# 	inputfile=open(filepath,'r+')
# 	feature=[]
# 	temp=[]
# 	for i in inputfile:
# 		i=list(eval(i))
# 		if len(temp)<2:
# 			temp.append(i)
# 		else:
# 			feature.append(temp)
# 			temp=[]
# 			temp.append(i)
# 	inputfile.close()	

# 	for i in feature:
# 		ppgx=i[0]
# 		ppgy=i[1]
		
# 		butterppgx=bandpass(2,5,200,ppgx)
# 		butterppgy=bandpass(2,5,200,ppgy)

# 		energy=calenergy(butterppgx)
# 		# plt.subplot(2,2,1)
# 		# plt.plot(range(len(ppgx)), ppgx, 'red')
# 		# plt.subplot(2,2,2)
# 		# plt.plot(range(len(ppgy)), ppgy, 'green')
# 		# plt.subplot(2,2,3)
# 		# plt.plot(range(len(butterppgx)), butterppgx, 'red')
# 		# plt.subplot(2,2,4)
# 		# plt.plot(range(len(butterppgy)), butterppgy, 'green')
# 		# plt.show()
# 		plt.subplot(3,1,1)
# 		plt.plot(range(len(butterppgx)), butterppgx, 'red')
# 		plt.subplot(3,1,2)
# 		plt.plot(range(len(energy)), energy, 'red')
# 		plt.subplot(3,1,3)
# 		plt.show()



dirpath='./oridata_3/tempuser_3/'
# dirpath='./oridata/sy_4/'
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


	butterppgx=bandpass(0.5,5,200,ppgx)
	butterppgy=bandpass(0.5,5,200,ppgy)
	energy=calenergy(butterppgx)
	JS=coarse_grained_detect(ppgx)

	plt.subplot(4,1,1)
	plt.plot(range(len(butterppgx)), butterppgx, 'red')
	plt.subplot(4,1,2)
	plt.plot(range(len(energy)), energy, 'red')
	plt.subplot(4,1,3)
	plt.plot(range(len(JS)), JS, 'red')
	# plt.ylim(0, 1)
	plt.subplot(4,1,4)
	plt.plot(range(len(ppgx)), ppgx, 'red')
	plt.xlabel(filepath)
	plt.show()