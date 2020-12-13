
# -*- coding=utf-8 -*-
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis,skew
from sklearn import preprocessing
from normal_tool import *
from scipy.stats import kurtosis,skew
from sklearn.decomposition import FastICA
import scipy.interpolate as spi

import random
import scipy as sc

from math import log


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                       #返回数据集的行数
    labelCounts = {}                                #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                         #对每组特征向量进行统计
        currentLabel = featVec[-1]                  #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1              #Label计数
    shannonEnt = 0.0                                #经验熵(香农熵)
    for key in labelCounts:                         #计算香农熵
        prob = float(labelCounts[key]) / numEntires #选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)           #利用公式计算
    return shannonEnt 

plt.rc('font',family='Times New Roman')

#定义向量的内积

def multiVector(A,B):
    
    C=sc.zeros(len(A))
    
    for i in range(len(A)):
        
        C[i]=A[i]*B[i]
        
    return sum(C)


#取定给定的反向的个数

def inVector(A,b,a):
    
    D=sc.zeros(b-a+1)
    
    for i in range(b-a+1):
        
        D[i]=A[i+a]
    
    return D[::-1]

 

#lMS算法的函数

def LMS(xn,dn,M,mu,itr):
    
    en=sc.zeros(itr)
    
    W=[[0]*M for i in range(itr)]
    
    
    for k in range(itr)[M-1:itr]:
        
        x=inVector(xn,k,k-M+1)
        d= x.mean()
        
        y=multiVector(W[k-1],x)
        
        en[k]=d -y
        
        W[k]=np.add(W[k-1],2*mu*en[k]*x) #跟新权重
    
    #求最优时滤波器的输出序列
    
    yn=sc.inf*sc.ones(len(xn))
    
    for k in range(len(xn))[M-1:len(xn)]:
    
        x=inVector(xn,k,k-M+1)
        
        yn[k]=multiVector(W[len(W)-1],x)
    
    return (yn,en)




def maline(data1,data2):	
	if abs(kurtosis(data1))>abs(kurtosis(data2)):#高峰度的是ma信号
		ma=data1
		pulse=data2
	else:
		ma=data2	
		pulse=data1
	return ma,pulse

def ppgfica(data1,data2):
	S = np.c_[data1, data2]
	ica = FastICA(n_components=2,algorithm='deflation')
	ica_dataset_X = ica.fit_transform(S)
	data1, data2=np.split(ica_dataset_X,[1],1)
	data1=data1.tolist()
	data1=[i[0] for i in data1]
	data2=data2.tolist()
	data2=[i[0] for i in data2]
	data1,data2=maline(data1,data2)
	return data1,data2

#导数序列 
def interationcal(data):
	interation=[]
	interation.append(data[1]-data[0])
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
	JS=[]
	for i in range(0,len(ppginter)-400):

		score1=array_distribute_cal(ppginter[i:i+200],alltag)
		score2=array_distribute_cal(ppginter[i+200:i+400],alltag)
	
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
	
def JS_incretempdata(data,incres):
	tempdata=[]
	for i in range(incres):
		tempdata.append(data[i])
	for i in data:
		tempdata.append(i)
	for i in range(incres):
		tempdata.append(data[-1-i])
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



def readdata(filepath):

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
	return ppgx,ppgy
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
def incretempdata(data,incre):
	tempdata=[]
	for i in range(incre):
		tempdata.append(data[0])
	for i in range(len(data)):
		tempdata.append(data[i])
	for i in range(incre):
		tempdata.append(data[-1])
	return tempdata		
def fine_grained_segment_4(dn,fre,top,bottom):
	pointstartindex=0
	pointendindex=0
	tag=0
	datalens=len(dn)
	tempdata=incretempdata(dn,int(fre/2))
	energy=[]
	for i in range(datalens):
		energy.append(np.std(tempdata[i:i+fre]))
	i=datalens-100
	while(i>fre):
		i=i-1
		if energy[i]>bottom:
			flag=0
			finalcount=0
			# 尾端半秒内小于阈值
			for j in range(100):
				if energy[i+j]<top:
					finalcount=finalcount+1
			if finalcount<80:
				flag=1
			if 0==flag:
				gesturecount=0
				for j in range(fre):
					if energy[i-j]>top:
						gesturecount=gesturecount+1
				if gesturecount<150:
					flag=1
			if 0==flag:
				t=i-150
				while t>2*fre:
					t=t-1
					if energy[t]<top:
						startcount=0
						for j in range(2*fre):
							if energy[t-j]<bottom:
								startcount=startcount+1
						if startcount>350:
							tag=1
							pointendindex=i
							pointstartindex=t
							break
		if tag==1:
			break		
	print(tag,pointstartindex,pointendindex)
	seglen=(pointstartindex-pointstartindex)								
	if seglen>400 :
		pointstartindex=0
		pointendindex=0
		tag=0
	return tag,pointstartindex,pointendindex



def paracal(data):
	butter=bandpass(2,5,200,data)
	energy=calenergy(butter)
	JS=coarse_grained_detect(energy)
	return butter,energy,JS



def paracal_2(data,start,end):
	butter=bandpass(2,5,200,data)
	energy=calenergy(butter)
	energy=energy[start:end]
	JS=coarse_grained_detect(energy)
	butter=butter[start:end]
	return butter,energy,JS


def paracal_3(data1,data2,start,end):
	data1=data1[start:end]
	data2=data2[start:end]
	butterx=bandpass(2,5,200,data1)
	buttery=bandpass(2,5,200,data2)
	icax,icay=ppgfica(butterx,buttery)

	energy=calenergy(butterx)
	JS=coarse_grained_detect(icax)
	# butterx=butterx[start:end]
	return butterx,energy,JS


def changehz(data,hz):

	newlens=len(data)*hz/200;
	print(newlens)
	newdata=[]
	for i in range(int(newlens)):
		newindex=int(i*200/hz)
		newdata.append(data[newindex])
	return newdata








filepath='./2020-10-11-21-25-54.csv' #[50:950]无手势段
appgx,appgy=readdata(filepath)

filepath='./2020-10-12-15-06-29.csv'#[550:1450]有手势段
bppgx,bppgy=readdata(filepath)

filepath='./2020-10-12-11-23-30.csv'#[650:1550]有手势段
cppgx,cppgy=readdata(filepath)








#gesture segmentation

# filepath='2020-08-30-17-30-19.csv'#[600:1500]有手势段
# ppgx,ppgy=readdata(filepath)
# butterx=bandpass(2,5,200,ppgx)
# buttery=bandpass(2,5,200,ppgy)
# butterx=butterx[400:1600]
# buttery=buttery[400:1600]

filepath='2020-09-13-09-25-10.csv'#[600:1500]有手势段
ppgx,ppgy=readdata(filepath)
ppgx,ppgy=readdata(filepath)
butterx=bandpass(2,5,200,ppgx)
buttery=bandpass(2,5,200,ppgy)
butterx=butterx[300:1500]
buttery=buttery[300:1500]




icax,icay=ppgfica(butterx,buttery)
tempdata=incretempdata(icax,100)
datalens=len(icax)
energy=[]
for i in range(datalens):
	energy.append(np.std(tempdata[i:i+200]))
tag,pointstartindex,pointendindex=fine_grained_segment_4(icax,200,0.04,0.03)
print(tag,pointstartindex,pointendindex)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.3)

# plt.subplot(3,1,3)
# plt.plot(range(len(butterx)), butterx, 'red',label="PPG_X",linewidth=0.6)
# # plt.plot(range(pointstartindex, pointendindex), butterx[pointstartindex:pointendindex], 'red',linewidth=2)
# plt.plot(range(len(buttery)), buttery, 'blue',label="PPG_Y",linewidth=0.6)
# # plt.plot(range(pointstartindex, pointendindex), buttery[pointstartindex: pointendindex], 'blue',linewidth=2)
# # plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
# plt.ylabel("PPG Reading")
# plt.xlabel("PPG Segment Index")
# plt.legend(loc = 'upper right')
plt.subplot(2,1,1)
# plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
plt.plot(range(len(icax)), icax, 'red',label="gesture",linewidth=0.6)
# plt.plot(range(pointstartindex, pointendindex), icax[pointstartindex:pointendindex], 'red',linewidth=2)
plt.ylabel("PPG Reading")
plt.xlabel("PPG Segment Index")
plt.legend(loc = 'upper right')
plt.subplot(2,1,2)
# plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
plt.plot(range(len(icay)), icay, 'blue',label="heart pulse",linewidth=0.6)
plt.ylabel("PPG Reading")
plt.xlabel("PPG Segment Index")
plt.legend(loc = 'upper right')
plt.show()


#具体的手势区间提取

# plt.subplot(3,1,1)
# plt.plot(range(len(butterx)), butterx, 'red',label="PPG_X",linewidth=0.6)
# plt.plot(range(pointstartindex, pointendindex), butterx[pointstartindex:pointendindex], 'red',linewidth=2)
# plt.plot(range(len(buttery)), buttery, 'blue',label="PPG_Y",linewidth=0.6)
# plt.plot(range(pointstartindex, pointendindex), buttery[pointstartindex: pointendindex], 'blue',linewidth=2)
# plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
# plt.ylabel("PPG Reading")
# plt.xlabel("PPG Segment Index")
# plt.legend(loc = 'upper right')

# plt.subplot(3,1,2)
# plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
# plt.plot(range(len(icay)), icay, 'blue',label="heart pulse",linewidth=0.6)
# plt.plot(range(len(icax)), icax, 'red',label="gesture",linewidth=0.6)
# plt.plot(range(pointstartindex, pointendindex), icax[pointstartindex:pointendindex], 'red',linewidth=2)
# plt.ylabel("PPG Reading")
# plt.xlabel("PPG Segment Index")
# plt.legend(loc = 'upper right')
# plt.subplot(3,1,3)

# plt.plot(range(len(energy)), energy, 'red',label="energy",linewidth=0.6)
# plt.axvline(pointstartindex,color="darkgreen",label="Starting Point",linestyle='-')
# plt.axvline(pointendindex,color="darkmagenta",label="Ending Point",linestyle='--')
# plt.axvspan(pointstartindex, pointendindex, facecolor='lightblue', alpha=0.5)
# plt.axhline(0.04,color="pink",label="Onset threshold",linestyle='--')
# plt.axhline(0.03,color="tan",label="Offset threshold",linestyle='--')
# plt.ylabel("Energy")
# plt.xlabel("Segment Index")
# plt.legend(loc = 'upper right')
# plt.show()
'''
'''
#gesture detection
# appgx,appgy=ppgfica(appgx,appgy)
# bppgx,bppgy=ppgfica(bppgx,bppgy)
# cppgx,cppgy=ppgfica(cppgx,cppgy)


# abutterppgx,aenergy,aJS=paracal_2(appgx,50,950)#无手势
# bbutterppgx,benergy,bJS=paracal_2(bppgx,550,1450)#有手势
# cbutterppgx,cenergy,cJS=paracal_2(cppgx,650,1550)#有手势


# ppgx=appgx[50:950]
# ppgy=appgy[50:950]
# ppgx=bppgx[100:900]
# ppgy=bppgy[100:900]

# ppgx=bppgx
# ppgy=bppgy
# ppgx=bppgx[550:1450]
# ppgy=bppgy[550:1450]

# ppgx=cppgx[650:1550]
# ppgy=cppgy[650:1550]

# # ppgx=cppgx
# # ppgy=cppgy

# butterx=bandpass(2,5,200,ppgx)
# buttery=bandpass(2,5,200,ppgy)
# t=minmaxscale(butterx)
# t=[round(i,2) for i in t]
# t=np.array(t)

# t=t.reshape(-1,1)
# print(calcShannonEnt(t))
# print(kurtosis(t))

# plt.subplot(2,1,1)
# plt.plot(range(len(butterx)), butterx, 'blue',linewidth=0.6)
# # plt.plot(range(len(buttery)), buttery, 'blue',linewidth=0.6)
# plt.show()

# icax,icay=ppgfica(butterx,buttery)

# x=[]
# y=[]
# for i in range(int(len(butterx)/20)):
# 	x.append(i*20)
# 	y.append(butterx[i*20])
# x=np.array(x)
# y=np.array(y)
# print(len(x))
# print(len(y))
# print(x)
# print(y)

# # x=np.arange(-np.pi,np.pi,1) #定义样本点X，从-pi到pi每次间隔1
# # y= np.sin(x)#定义样本点Y，形成sin函数
# # print(x)
# # print(y)
# # print(len(x))
# ipo3=spi.splrep(x,y,k=3) #样本点导入，生成参数
# iy3=spi.splev(range(len(butterx)),ipo3)

# d=[]
# for i in range(len(butterx)):
# 	d.append(abs(butterx[i]-iy3[i]))

# plt.subplot(5,1,1)
# plt.plot(range(len(ppgx)), ppgx, 'red',linewidth=0.6)
# plt.plot(range(len(ppgy)), ppgy, 'blue',linewidth=0.6)
# plt.subplot(5,1,2)
# plt.plot(range(len(butterx)), butterx, 'red',linewidth=0.6)
# plt.plot(range(len(buttery)), buttery, 'blue',linewidth=0.6)
# plt.plot(range(len(butterx)), iy3, 'green',linewidth=0.6)
# plt.subplot(5,1,3)
# plt.plot(range(len(icax)), icax, 'red',linewidth=0.6)
# plt.plot(range(len(icay)), icay, 'blue',linewidth=0.6)
# plt.subplot(5,1,4)

# xenergy=calenergy(icax)
# yenergy=calenergy(icay)
# plt.plot(range(len(xenergy)), xenergy, 'red',linewidth=0.6)
# plt.plot(range(len(yenergy)), yenergy, 'blue',linewidth=0.6)
# plt.subplot(5,1,5)
# # plt.plot(x, y, 'red',linewidth=0.6)
# plt.plot(range(len(d)), d, 'red',linewidth=0.6)

# # xJS=coarse_grained_detect(icax)
# # yJS=coarse_grained_detect(icay)
# # plt.plot(range(len(xJS)), xJS, 'red',linewidth=0.6)
# # plt.plot(range(len(yJS)), yJS, 'blue',linewidth=0.6)

# plt.show()

'''

abutterppgx,aenergy,aJS=paracal_3(appgx,appgy,50,950)#无手势
bbutterppgx,benergy,bJS=paracal_3(bppgx,bppgy,550,1450)#有手势
cbutterppgx,cenergy,cJS=paracal_3(cppgx,cppgy,650,1550)#有手势


plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.3)

plt.subplot(3,1,1)
plt.plot(range(len(bbutterppgx)), bbutterppgx, 'red',label='session 1, hava gesture',linewidth=0.6)
plt.plot(range(495, 660), bbutterppgx[495:660], 'red',linewidth=2)
plt.plot(range(len(cbutterppgx)), cbutterppgx, 'blue',label='session 2, have gesture',linewidth=0.6)
plt.plot(range(397, 581), cbutterppgx[397: 581], 'blue',linewidth=2)
plt.plot(range(len(abutterppgx)), abutterppgx, 'green',label='session 2, no gesture',linewidth=0.6)

plt.ylabel("PPG Reading")
plt.xlabel("PPG Segment Index")
plt.legend(loc = 'upper right')

plt.subplot(3,1,2)
plt.plot(range(len(benergy)), benergy, 'red',linewidth=0.6,label='session 1, hava gesture')
plt.plot(range(len(cenergy)), cenergy, 'blue',linewidth=0.6,label='session 2, have gesture')
plt.plot(range(len(aenergy)), aenergy, 'green',linewidth=0.6,label='session 2, no gesture')
# plt.plot([0, 900], [1600, 1600], c='black', linestyle='--')
# plt.ylim(1000, 6000)
plt.ylabel("Energy")
plt.xlabel("Segment Index")
plt.legend(loc = 'upper right')

plt.subplot(3,1,3)
plt.plot(range(len(bJS)), bJS, 'red',linewidth=0.6,label='session 1, have gesture')
plt.plot(range(len(cJS)), cJS, 'blue',linewidth=0.6,label='session 2, have gesture')
plt.plot(range(len(aJS)), aJS, 'green',linewidth=0.6,label='session 2, no gesture')
plt.plot([0, 900], [0.15, 0.15], c='black', linestyle='--')
plt.ylim(0, 0.3)
plt.ylabel("JS divergence")
plt.xlabel("Segment Index")
plt.legend(loc = 'upper right')
plt.show()


'''
'''

#不同频率的ppg信号图和采样频率的影响
filepath='./2020-10-12-15-06-29.csv'#[550:1450]有手势段
bppgx,bppgy=readdata(filepath)
ppg=bppgx
ppg=ppg[400:]
ppg_100=changehz(ppg,100)
ppg_20=changehz(ppg,20)
ppg_10=changehz(ppg,10)

butter=bandpass(2,4,200,ppg)
butter_100=bandpass(2,4,100,ppg_100)
butter_20=bandpass(2,4,20,ppg_20)
# butter_10=bandpass(2,4,10,ppg_10)
butter_10=highpass(1,10,ppg_10)
plt.figure(1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.3)
plt.subplot(2,2,1)
plt.plot(range(len(butter)), butter, 'red',label='200hz',linewidth=0.6)
plt.ylabel("PPG Reading")
plt.xlabel("Segment Index")
plt.legend(loc = 'upper right')
plt.subplot(2,2,3)
plt.plot(range(len(butter_20)), butter_20, 'blue',label='20hz',linewidth=0.6)
plt.ylabel("PPG Reading")
plt.xlabel("Segment Index")
plt.legend(loc = 'upper right')
plt.subplot(1,2,2)

x=["20","40","60","80","100","150","200"]
y=[89.12,88.38,89.33,88.8,89.44,89,89.14]
color=['#00BF00',"#00BF10","#00BF20","#00BF30","#00BF40","#00BF50","#00BF60"]
plt.bar(x, y, color=color,alpha=0.7)
plt.ylim(0, 100)
plt.yticks(range(0,101,10))
plt.ylabel("Accuracy (%)")
plt.xlabel("Sampling Rate (Hz)")

plt.show()




'''
'''
#9个手势的FAR和FRR图
x=[i for i in range(1,10)]
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None,
                wspace=0.3, hspace=0.2)

plt.subplot(1,2,2)
y=[8.95,11.56,9.07,11.01,12.87,8.92,11.78,8.45,9.25]
color=['#800000','#800010','#800020','#800030','#800040','#800050','#800060','#800070','#800080','#800090']
plt.bar(x, y,color=color,alpha=0.8,edgecolor='black')
plt.axhline(10.03,color="darkorange",label="Acerage of gesture: 10.03%",linestyle='-')
plt.ylim(0, 30)
plt.xticks(range(1,10))
# plt.ylabel("False Accept Rate (%)")
plt.ylabel("Error Equral Rate (%)")
plt.xlabel("Gesture Index")
plt.legend(loc = 'center')
# plt.subplot(1,3,2)
# y=[8.4,10.71,8.87,10.36,11.33,8.19,11.01,7.895,8.58]
# color=['#FFD700','#FFD710','#FFD720','#FFD730','#FFD740','#FFD750','#FFD760','#FFD770','#FFD780','#FFD790']
# plt.bar(x, y,color=color,alpha=0.8,edgecolor='black')
# plt.ylim(0, 100)
# plt.xticks(range(1,10))
# plt.ylabel("False Reject Rate (%)")
# plt.xlabel("Gesture Index")

plt.subplot(1,2,1)
y=[91.32,88.86,90.79,89.3,87.58,91.43,88.59,91.82,91.08]
color=['#191900','#191910','#191920','#191930','#191940','#191950','#191960','#191970','#191980','#191990']
plt.bar(x, y,color=color,alpha=0.8,edgecolor='black')
plt.ylim(0, 100)
plt.axhline(89.83,color="red",label="Acerage of gesture: 89.83%",linestyle='-')

plt.xticks(range(1,10))
plt.ylabel("Accuracy (%)")
plt.xlabel("Gesture Index")
plt.legend(loc = 'center')

plt.show()

'''
'''


#不同算法对注册样本数的需要程度
x=[i for i in range(1,13)]
x=np.arange(1,13)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=0.2)

ds=[90.56,92.97,94.55,94.06,93.53,93.97,94.38,94.25,94.05,93.85,94.72,93.27]

ss=[86.87,88.88,89.30,89.51,89.69,90.56,90.51,90.6,90.92,91.22,91.2,89.35]

GBT=[59.72,67.96,75,78.84,81.93,83.69,85.78,86.67,86.84,87.65,88.01,88.15]

SVM=[34.86,45.84,70.44,79.66,87.78,88.82,90.67,89.95,89.89,89.97,90.25,89.87]

plt.plot(x, GBT, 'darkorange',label='GBT [8]',linewidth=0.6)
plt.plot(x, SVM, 'navy',label='SVM [2]',linewidth=0.6)
plt.plot(x, ss, 'limegreen',label='Siamese Network',linewidth=0.6)
plt.plot(x, ds, 'red',label='Double Siamese Network',linewidth=0.6)

# width = 0.2
# plt.bar(x+width-0.1, GBT,  width=width, label='GBT [8]',color='darkorange')
# plt.bar(x+2*width-0.1, SVM,  width=width, label='SVM [2]',color='navy')
# plt.bar(x-width+0.1, ss,  width=width, label='Based Siamese Network [10] [34]',color='limegreen')
# plt.bar(x-2*width+0.1, ds,  width=width, label='Double Siamese Network',color='red')

# width = 0.2
# plt.bar(x, GBT,  width=width, label='GBT [8]',color='darkorange')
# plt.bar(x+width, SVM,  width=width, label='SVM [2]',color='navy')
# plt.bar(x-width, ds,  width=width, label='Double Siamese Network',color='red')

plt.ylim(20, 100)
plt.xticks(range(1,13))
plt.ylabel("Accuracy (%)")
plt.xlabel("Register Sample ")
plt.legend(loc = 'lower right')
plt.show()
'''
'''
#训练集的志愿者数量的影响

# x=[str(i) for i in range(2,37,2)]
x=[i for i in range(2,37,2)]
y=[80.48,83.8,87.25,88.08,86.14,88.29,89.63,89.4,88.19,88.3,87.69,90.87,90.82,90.32,90.03,90.1,90.2,90.51]
std=[11.21,9.31,7.02,7.08,5.39,8.06,6.22,7.52,5.63,6.6,6.33,6.98,5.22,5.65,5.54,6.2,6.12,5.16]
y=np.array(y)
std=np.array(std)

# plt.plot(x, y, color='red',linewidth=0.6)
# plt.fill_between(x, y+std, y-std, color='lightblue', alpha=0.4)
plt.errorbar(x,y,yerr=std,fmt='^r',ecolor='blue',elinewidth=1,capsize=3)
plt.plot(x, y, color='red',marker='^',markerfacecolor ='red',linewidth=0.6)
plt.ylim(60, 100)
plt.ylabel("Accuracy (%)")
plt.xlabel("The Number of Volunteer")

plt.show()

'''

'''
#ROC曲线图
spmx=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003524672708962739, 0.003524672708962739, 0.006042296072507553, 0.006545820745216516, 0.008559919436052367, 0.008559919436052367, 0.009566968781470292, 0.013091641490433032, 0.01661631419939577, 0.019133937562940583, 0.022658610271903322, 0.026686807653575024, 0.03272910372608258, 0.03474320241691843, 0.042799597180261835, 0.045317220543806644, 0.050352467270896276, 0.061933534743202415, 0.07250755287009064, 0.08559919436052367, 0.09667673716012085, 0.1067472306143001, 0.12890231621349446, 0.15357502517623364, 0.17623363544813697, 0.19889224572004027, 0.22104733131923465, 0.24370594159113795, 0.26586102719033233, 0.29154078549848944, 0.31419939577039274, 0.3393756294058409, 0.364551863041289, 0.3877139979859013, 0.4108761329305136, 0.4375629405840886, 0.46273917421953675, 0.4909365558912387, 0.5125881168177241, 0.5342396777442094, 0.5558912386706949, 0.5795568982880162, 0.6002014098690835, 0.6238670694864048, 0.6455186304128903, 0.6676737160120846, 0.6878147029204431, 0.7079556898288016, 0.728600201409869, 0.7497482376636455, 0.770392749244713, 0.7910372608257804, 0.8126888217522659, 0.8358509566968781, 0.8559919436052367, 0.8766364551863042, 0.8997985901309165, 0.9199395770392749, 0.9436052366565961, 0.9642497482376636, 0.9843907351460222]
spmy=[0.9748237663645518, 0.9425981873111783, 0.9164149043303121, 0.8741188318227593, 0.8439073514602216, 0.8167170191339376, 0.7875125881168177, 0.7663645518630413, 0.7371601208459214, 0.6998992950654582, 0.6636455186304129, 0.6334340382678751, 0.6092648539778449, 0.5740181268882175, 0.540785498489426, 0.5105740181268882, 0.48036253776435045, 0.4521651560926485, 0.4259818731117825, 0.39375629405840884, 0.36656596173212486, 0.3403826787512588, 0.3162134944612286, 0.29103726082578046, 0.2598187311178248, 0.2366565961732125, 0.21651560926485397, 0.18731117824773413, 0.16314199395770393, 0.1419939577039275, 0.11983887210473314, 0.0986908358509567, 0.07854984894259819, 0.059415911379657606, 0.04229607250755287, 0.03121852970795569, 0.02014098690835851, 0.01812688821752266, 0.016112789526686808, 0.012084592145015106, 0.008056394763343404, 0.007049345417925478, 0.007049345417925478, 0.006042296072507553, 0.004028197381671702, 0.002014098690835851, 0.002014098690835851, 0.002014098690835851, 0.002014098690835851, 0.0010070493454179255, 0.0010070493454179255, 0.0010070493454179255, 0.0010070493454179255, 0.0010070493454179255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

spx=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004187604690117253, 0.0008375209380234506, 0.0008375209380234506, 0.0008375209380234506, 0.001256281407035176, 0.001256281407035176, 0.002931323283082077, 0.0046063651591289785, 0.00628140703517588, 0.00711892797319933, 0.011725293132328308, 0.012981574539363484, 0.01507537688442211, 0.01968174204355109, 0.021775544388609715, 0.02763819095477387, 0.0347571189279732, 0.04983249581239531, 0.06658291457286432, 0.08835845896147404, 0.10971524288107203, 0.12981574539363483, 0.14865996649916247, 0.16624790619765495, 0.18676716917922948, 0.20896147403685092, 0.22906197654941374, 0.2516750418760469, 0.2721943048576214, 0.29229480737018426, 0.314070351758794, 0.33500837520938026, 0.3555276381909548, 0.37646566164154105, 0.3969849246231156, 0.41792294807370184, 0.4380234505862647, 0.46063651591289784, 0.48199329983249584, 0.5058626465661642, 0.5284757118927973, 0.5494137353433836, 0.5707705192629816, 0.5917085427135679, 0.6130653266331658, 0.6373534338358459, 0.6582914572864321, 0.6804857621440537, 0.7030988274706867, 0.724036850921273, 0.7445561139028476, 0.7680067001675042, 0.791038525963149, 0.8136515912897823, 0.8350083752093802, 0.8555276381909548, 0.8756281407035176, 0.8957286432160804, 0.9166666666666666, 0.9371859296482412, 0.957286432160804, 0.9773869346733668]
spy=[0.9731993299832495, 0.9455611390284757, 0.9154103852596315, 0.88107202680067, 0.8509212730318257, 0.8056951423785594, 0.7671691792294807, 0.7294807370184254, 0.7093802345058626, 0.6758793969849246, 0.6465661641541038, 0.6113902847571189, 0.5770519262981575, 0.5469011725293133, 0.5234505862646566, 0.4983249581239531, 0.4715242881072027, 0.44472361809045224, 0.4221105527638191, 0.3961474036850921, 0.3676716917922948, 0.33752093802345057, 0.3157453936348409, 0.29229480737018426, 0.26716917922948075, 0.24706867671691793, 0.2236180904522613, 0.2018425460636516, 0.1800670016750419, 0.15829145728643215, 0.1373534338358459, 0.12897822445561138, 0.11306532663316583, 0.09296482412060302, 0.07202680067001675, 0.05778894472361809, 0.049413735343383586, 0.04355108877721943, 0.03433835845896147, 0.02847571189279732, 0.022613065326633167, 0.019262981574539362, 0.01256281407035176, 0.010050251256281407, 0.010050251256281407, 0.008375209380234505, 0.007537688442211055, 0.006700167504187605, 0.006700167504187605, 0.006700167504187605, 0.0041876046901172526, 0.0033500837520938024, 0.0033500837520938024, 0.002512562814070352, 0.0016750418760469012, 0.0008375209380234506, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

smx=[0.0004906771344455348, 0.0009813542688910696, 0.002944062806673209, 0.003925417075564278, 0.007360157016683023, 0.008832188420019628, 0.009322865554465163, 0.012757605495583905, 0.017664376840039256, 0.018645731108930325, 0.022571148184494603, 0.025024533856722278, 0.02845927379784102, 0.03680078508341511, 0.04416094210009813, 0.04759568204121688, 0.05053974484789009, 0.05593719332679097, 0.05937193326790972, 0.06231599607458292, 0.0662414131501472, 0.07556427870461237, 0.08537782139352307, 0.09371933267909716, 0.10206084396467124, 0.10942100098135427, 0.1182531894013739, 0.12855740922473013, 0.13297350343473993, 0.14327772325809618, 0.15554465161923453, 0.1633954857703631, 0.176153091265947, 0.1859666339548577, 0.19381746810598627, 0.20314033366045142, 0.21589793915603533, 0.23748773307163887, 0.260549558390579, 0.2831207065750736, 0.3042198233562316, 0.32433758586849853, 0.34445534838076547, 0.36702649656526004, 0.38959764474975467, 0.41462217860647693, 0.43719332679097156, 0.4685966633954858, 0.4896957801766438, 0.5166830225711482, 0.5412168792934249, 0.5691854759568205, 0.5937193326790972, 0.6172718351324828, 0.6373895976447498, 0.6619234543670265, 0.68351324828263, 0.703631010794897, 0.7291462217860648, 0.7536800785083415, 0.7796859666339548, 0.8071638861629048, 0.8316977428851815, 0.8640824337585868, 0.8910696761530913, 0.915603532875368, 0.9357212953876349, 0.9592737978410206, 0.9823356231599607]
smy=[0.9793915603532876, 0.9587831207065751, 0.9322865554465162, 0.9106967615309126, 0.8900883218842002, 0.8655544651619235, 0.8439646712463199, 0.8174681059862611, 0.7929342492639843, 0.7684003925417076, 0.7379784102060843, 0.7134445534838076, 0.6830225711481845, 0.6526005888125613, 0.6211972522080471, 0.5966633954857704, 0.5731108930323847, 0.55053974484789, 0.5279685966633955, 0.5014720314033366, 0.48086359175662413, 0.45436702649656524, 0.4308145240431796, 0.4072620215897939, 0.3788027477919529, 0.35328753680078506, 0.3267909715407262, 0.30520117762512267, 0.2787046123650638, 0.25515210991167814, 0.2325809617271835, 0.2070657507360157, 0.18645731108930325, 0.16584887144259078, 0.14327772325809618, 0.12168792934249265, 0.10107948969578018, 0.08439646712463199, 0.07850834151128558, 0.06673209028459273, 0.05789990186457311, 0.05103042198233562, 0.03729146221786065, 0.0323846908734053, 0.03140333660451423, 0.02845927379784102, 0.02845927379784102, 0.02747791952894995, 0.02747791952894995, 0.02551521099116781, 0.02551521099116781, 0.023552502453385672, 0.022571148184494603, 0.020608439646712464, 0.020608439646712464, 0.018645731108930325, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.017664376840039256, 0.016683022571148183, 0.016683022571148183, 0.016683022571148183, 0.016683022571148183]

dspmx= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.004373177842565598, 0.0058309037900874635, 0.0058309037900874635, 0.0058309037900874635, 0.007288629737609329, 0.008746355685131196, 0.008746355685131196, 0.008746355685131196, 0.01020408163265306, 0.01020408163265306, 0.016034985422740525, 0.01749271137026239, 0.030612244897959183, 0.052478134110787174, 0.06705539358600583, 0.09037900874635568, 0.12099125364431487, 0.14285714285714285, 0.16326530612244897, 0.18658892128279883, 0.20845481049562684, 0.22886297376093295, 0.24927113702623907, 0.27696793002915454, 0.30466472303207, 0.3294460641399417, 0.35131195335276966, 0.3746355685131195, 0.3965014577259475, 0.41690962099125367, 0.44752186588921283, 0.47959183673469385, 0.5029154518950437, 0.5233236151603499, 0.543731778425656, 0.5641399416909622, 0.5860058309037901, 0.607871720116618, 0.6282798833819242, 0.6486880466472303, 0.6705539358600583, 0.6909620991253644, 0.7113702623906706, 0.7317784256559767, 0.7521865889212828, 0.7784256559766763, 0.7988338192419825, 0.8192419825072886, 0.8396501457725948, 0.8658892128279884, 0.8877551020408163, 0.9154518950437318, 0.9373177842565598, 0.9577259475218659, 0.9795918367346939, 1.0]
dspmy=[0.9766763848396501, 0.956268221574344, 0.9271137026239067, 0.8979591836734694, 0.8775510204081632, 0.8396501457725948, 0.8134110787172012, 0.7900874635568513, 0.7667638483965015, 0.7346938775510204, 0.7142857142857143, 0.6909620991253644, 0.6705539358600583, 0.641399416909621, 0.6151603498542274, 0.5918367346938775, 0.5568513119533528, 0.5364431486880467, 0.5160349854227405, 0.49271137026239065, 0.4606413994169096, 0.43440233236151604, 0.4139941690962099, 0.3877551020408163, 0.3556851311953353, 0.33527696793002915, 0.30612244897959184, 0.27988338192419826, 0.2478134110787172, 0.22448979591836735, 0.19825072886297376, 0.17784256559766765, 0.15743440233236153, 0.13411078717201166, 0.11370262390670553, 0.09329446064139942, 0.07580174927113703, 0.05539358600583091, 0.04664723032069971, 0.04081632653061224, 0.023323615160349854, 0.01749271137026239, 0.011661807580174927, 0.0058309037900874635, 0.0058309037900874635, 0.0058309037900874635, 0.0058309037900874635, 0.0058309037900874635, 0.0058309037900874635, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0029154518950437317, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

spmx=np.array(spmx)*100
spmy=np.array(spmy)*100
spx=np.array(spx)*100
spy=np.array(spy)*100
smx=np.array(smx)*100
smy=np.array(smy)*100
dspmx=np.array(dspmx)*100
dspmy=np.array(dspmy)*100

plt.plot(spx,spy,color='blue',marker='s',label='Siamese Network with PPG')
plt.plot(smx,smy,color='green',marker='D',label='Siamese Network with Motion')
plt.plot(spmx,spmy,color='red',marker='v',label='Siamese Network with PPG and Motion')
plt.plot(dspmx,dspmy,color='black',marker='^',label='Double Siamese Network with PPG and Motion')
plt.ylabel("False Reject Rate (%)")
plt.xlabel("False Accept Rate (%)")
plt.legend(loc = 'upper right')

plt.show()


'''



# plt.subplot(2,2,1)
# plt.plot(range(len(butter)), butter, 'red',label='200hz',linewidth=0.6)
# plt.legend(loc = 'upper right')
# plt.subplot(2,2,2)
# plt.plot(range(len(butter_100)), butter_100, 'red',label='100hz',linewidth=0.6)
# plt.legend(loc = 'upper right')
# plt.subplot(2,2,3)
# plt.plot(range(len(butter_20)), butter_20, 'red',label='20hz',linewidth=0.6)
# plt.legend(loc = 'upper right')
# plt.subplot(2,2,4)
# plt.plot(range(len(butter_10)), butter_10, 'red',label='10hz',linewidth=0.6)
# plt.legend(loc = 'upper right')
# plt.show()



# plt.subplot(3,1,1)
# plt.plot(range(len(bbutterppgx)), bbutterppgx, 'red')

# plt.axvspan(495, 660, facecolor='black', alpha=0.5)
# plt.subplot(3,1,2)
# plt.plot(range(len(abutterppgx)), abutterppgx, 'blue')
# plt.subplot(3,1,3)
# plt.plot(range(len(cbutterppgx)), cbutterppgx, 'green')
# plt.axvspan(397, 581, facecolor='black', alpha=0.5)
# plt.show()


# plt.subplot(2,1,1)
# plt.plot(range(len(aenergy)), aenergy, 'red')
# plt.plot(range(len(benergy)), benergy, 'blue')
# plt.plot(range(len(cenergy)), cenergy, 'green')
# plt.axhline(y=1600,color="black")
# plt.subplot(2,1,2)
# plt.plot(range(len(aJS)), aJS, 'red')
# plt.plot(range(len(bJS)), bJS, 'blue')
# plt.plot(range(len(cJS)), cJS, 'green')
# plt.axhline(y=0.15,color="black")
# plt.ylim(0, 0.3)
# plt.show()

# lens=range(495, 660,1)
# lens=range(len(abutterppgx))
# print(lens)


# cbutterppgx,aenergy,aJS=paracal(appgx)

# aenergy=aenergy[50:950]
# abutterppgx=abutter[50:950]
# aJS=aJS[50:950]

# cbutterppgx,benergy,bJS=paracal(bppgx)

# benergy=benergy[550:1450]
# bbutterppgx=bbutter[550:1450]
# bJS=bJS[550:1450]

# cbutterppgx,cenergy,cJS=paracal(cppgx)

# cenergy=cenergy[650:1550]
# cbutterppgx=cbutterppgx[650:1550]
# cJS=cJS[650:1550]



# plt.subplot(3,1,1)
# plt.plot(range(len(butterppgx)), butterppgx, 'red')
# plt.subplot(3,1,2)
# plt.plot(range(len(energy)), energy, 'red')
# plt.subplot(3,1,3)
# plt.plot(range(len(JS)), JS, 'red')
# plt.ylim(0, 0.3)
# plt.xlabel(filepath)
# plt.show()


# filepath='./2020-10-12-15-06-29.csv'#粗粒度处理不够明显
# filepath='./2020-10-05-21-22-45.csv'







# filepath='./2020-10-05-20-58-14.csv' #[50:950]干净手势段
# filepath='./2020-10-14-19-10-15.csv' #[50:950]干净手势段

# import padasip as pa 

#d带噪音的信号，v噪音，x原信号，y输出值，错误值，权重值，v要转化为单长为1的序列
# f = pa.filters.FilterLMS(n=1, mu=0.1, w="random")

# d=butterx[900:1500]
# v=butterx[200:800]
# v=np.array(v)
# v=v.reshape(len(v),1)
# y, e, w = f.run(d, v)
# # (yn1,en1)=LMS(bppgx[900:1500],bppgx[200:800],4,0.00001,4)
# # yn1=bandpass(2,5,200,yn1)

# # (yn2,en2)=LMS(buttery,icay,1,0.01,80)
# # print(yn1)


