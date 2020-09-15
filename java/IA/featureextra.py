# -*- coding=utf-8 -*-
import numpy as np
from pywt import wavedec
import IAtool
from statsmodels.tsa.stattools import acf
from dtw import dtw
from scipy import stats
import math
import pandas as pd



manhattan_distance = lambda x, y: np.abs(x - y)

#计算样本之间的相dtw，选择最能代表模型的3个样本
def modelselect(segment):
	selen=len(segment)
	distence=[[0 for i in range(selen)] for i in range(selen)]
	for i in range(selen):
		for j in range(i,selen):
			if i==j:
				distence[i][i]=0
			else:
				d, cost_matrix, acc_cost_matrix, path = dtw(segment[i], segment[j], dist=manhattan_distance)
				distence[i][j]=d
				distence[j][i]=d
	# print(distence)			
	#求不同模块之间的相互平均距离
	alldistence=[]		
	for i in range(selen):
		# print(distence[i])
		# print(np.mean(distence[i]))
		alldistence.append(np.mean(distence[i]))
	# print(alldistence)
	dtwmodel=[]
	tempalldistence=sorted(alldistence)
	# print(tempalldistence)
	#将最能代表动作的3个模块添加到model里
	dtwmodel.append(segment[alldistence.index(tempalldistence[0])])
	dtwmodel.append(segment[alldistence.index(tempalldistence[1])])
	dtwmodel.append(segment[alldistence.index(tempalldistence[2])])
	# print("dtwmodel:",dtwmodel)
	return dtwmodel


#ppgpass方案的特征选择
def ppgpassfeatureextra(data,fre,dtwmodel):
	data=np.array(data)
	feature=[]
	# Time Domain
	feature.append(np.mean(data))
	feature.append(np.std(data))
	rms=math.sqrt(sum([x ** 2 for x in data]) / len(data))
	feature.append(rms)
	ptp=np.max(data)-np.min(data)
	feature.append(ptp)
	
	#相关系数和动态时间规整
	for i in range(len(dtwmodel)):
		tempdtw=[]
		tempcross=[]
		for j in range(len(dtwmodel[i])):
			d, cost_matrix, acc_cost_matrix, path = dtw(data, dtwmodel[i][j], dist=manhattan_distance)
			tempdtw.append(d)

			gdata = pd.Series(data)
			gmodel = pd.Series(dtwmodel[i][j])
			crosscor=gdata.corr(gmodel)
			tempcross.append(crosscor)

		dtwscore=np.mean(tempdtw)
		feature.append(dtwscore)	
		crosssscore=np.mean(tempcross)
		feature.append(crosssscore)	

	#Frequency Domain
	freqs, datafft=IAtool.fft(data,fre)
	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(datafft[i])
		else:
			break
	# print(extrafft)		
	feature.append(stats.skew(extrafft))
	feature.append(stats.kurtosis(extrafft))
	feature.append(np.mean(extrafft))
	feature.append(np.median(extrafft))
	feature.append(np.std(extrafft))
	ptp=np.max(extrafft)-np.min(extrafft)
	feature.append(ptp)

	#time-frequence

	#Discrete Wavelet Transform
	coeffs = wavedec(data, 'haar', level=3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))
	feature.append(np.std(cA3))
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	feature.append(rms)
	ptp=np.max(cA3)-np.min(cA3)
	feature.append(ptp)

	#Wigner Ville Distribution
	x,y,z=IAtool.WignerVillecal(data)
	feature.append(np.max(z))
	feature.append(np.min(z))
	feature.append(np.max(z)-np.min(z))
	feature.append(np.std(z))
	maxindex=z.index(np.max(z))
	# maxindex=z.index(np.max(z))
	feature.append(x[maxindex])
	feature.append(y[maxindex])

	#Autoregressive Coefficients
	result=acf(data)
	for i in range(9):
		feature.append(result[i])		
	return feature

#特征原集
def orifeatureextra(data):
	feature=[]
	# print("时域特征提取")
	feature.append(np.mean(data))
	# feature.append(np.std(data))
	feature.append(math.sqrt(sum([x ** 2 for x in data-np.mean(data)]) / (len(data)-1)))
	feature.append(np.max(data)-np.min(data))
	feature.append(np.max(data))
	feature.append(np.min(data))
	feature.append(np.median(data))
	feature.append(stats.kurtosis(data))
	feature.append(stats.skew(data))

	mean=np.mean(data)
	rms=math.sqrt(sum([x ** 2 for x in data]) / len(data))
	amplitude=sum([abs(i-mean) for i in data])/len(data)
	diversion=sum([abs(i) for i in data])/len(data)

	feature.append(rms)
	feature.append(amplitude)
	feature.append(diversion)

	interval=[]
	for i in range(len(data)-1):
		interval.append(data[i+1]-data[i])

	feature.append(np.max(interval))
	feature.append(np.min(interval))
	feature.append(stats.kurtosis(interval))
	feature.append(stats.skew(interval))
	feature.append(np.median(interval))

	mean=np.mean(interval)
	rms=math.sqrt(sum([x ** 2 for x in interval]) / len(interval))
	amplitude=sum([abs(i-mean) for i in interval])/len(interval)
	diversion=sum([abs(i) for i in interval])/len(interval)
	feature.append(rms)
	feature.append(amplitude)
	feature.append(diversion)




	# print("频域特征提取")

	#Frequency Domain
	fre=200
	freqs, datafft=IAtool.fft(data,fre)

	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(datafft[i])
		else:
			break
	# print(extrafft)		

	feature.append(np.mean(extrafft))
	feature.append(math.sqrt(sum([x ** 2 for x in extrafft-np.mean(extrafft)]) / (len(extrafft)-1)))
	feature.append(np.max(extrafft)-np.min(extrafft))
	feature.append(np.max(extrafft))
	feature.append(np.min(extrafft))
	feature.append(np.median(extrafft))
	feature.append(stats.kurtosis(extrafft))
	feature.append(stats.skew(extrafft))
	# print("时频域特征提取")

	#Discrete Wavelet Transform
	coeffs = wavedec(data, 'haar', level=3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))
	feature.append(math.sqrt(sum([x ** 2 for x in cA3-np.mean(cA3)]) / (len(cA3)-1)))
	feature.append(np.max(cA3)-np.min(cA3))

	mean=np.mean(interval)
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	amplitude=sum([abs(i-mean) for i in cA3])/len(cA3)
	diversion=sum([abs(i) for i in cA3])/len(cA3)
	feature.append(rms)
	feature.append(amplitude)
	feature.append(diversion)


	#Wigner Ville Distribution
	# x,y,z=IAtool.WignerVillecal(data)
	
	# feature.append(np.max(z))
	# feature.append(np.min(z))
	# feature.append(np.max(z)-np.min(z))
	# feature.append(np.std(z))
	# maxindex=z.index(np.max(z))
	# # maxindex=z.index(np.max(z))
	# feature.append(x[maxindex])
	# feature.append(y[maxindex])

	#Autoregressive Coefficients
	result=acf(data)
	for i in range(1,10):
		feature.append(result[i])

	return feature


def featureextra(data1,data2):
	data1=np.array(data1)
	data2=np.array(data2)

	#gesture feature
	feature=[]
	# print("时域特征提取")
	feature.append(np.mean(data1))
	feature.append(np.std(data1))
	feature.append(rms)
	ptp=np.max(data1)-np.min(data1)
	feature.append(ptp)
	feature.append(np.max(data1))
	feature.append(np.min(data1))
	feature.append(stats.skew(data1))
	feature.append(stats.kurtosis(data1))
	feature.append(np.median(data1))

	rms=math.sqrt(sum([x ** 2 for x in data1]) / len(data1))
	mean=np.mean(data1)
	amplitude=sum([abs(i-mean) for i in data1])/len(data1)
	feature.append(amplitude)
	diversion=sum([abs(i) for i in data1])/len(data1)
	feature.append(diversion)

	interval=[]
	for i in range(len(data1)-1):
		interval.append(data1[i+1]-data1[i])
	feature.append(np.max(interval))
	feature.append(np.min(interval))

	mean=np.mean(interval)
	amplitude=sum([abs(i-mean) for i in interval])/len(interval)
	feature.append(amplitude)

	diversion=sum([abs(i) for i in interval])/len(interval)
	feature.append(diversion)

	feature.append(stats.skew(interval))
	feature.append(stats.kurtosis(interval))
	feature.append(np.median(interval))



	# print("频域特征提取")

	#Frequency Domain
	fre=200
	freqs, data1fft=IAtool.fft(data1,fre)

	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(data1fft[i])
		else:
			break
	feature.append(np.mean(extrafft))
	feature.append(np.std(extrafft))
	feature.append(np.max(extrafft)-np.min(extrafft))
	feature.append(np.max(extrafft))
	feature.append(np.min(extrafft))
	feature.append(np.median(extrafft))
	feature.append(stats.kurtosis(extrafft))
	feature.append(stats.skew(extrafft))

	# print("时频域特征提取")

	#Discrete Wavelet Transform
	coeffs = wavedec(data1, 'haar', level=3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))
	feature.append(np.std(cA3))
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	feature.append(rms)
	ptp=np.max(cA3)-np.min(cA3)
	feature.append(ptp)

	#Wigner Ville Distribution
	x,y,z=IAtool.WignerVillecal(data1)

	feature.append(np.max(z))
	feature.append(np.min(z))
	feature.append(np.max(z)-np.min(z))
	feature.append(np.std(z))
	maxindex=z.index(np.max(z))
	# maxindex=z.index(np.max(z))
	feature.append(x[maxindex])
	feature.append(y[maxindex])

	#Autoregressive Coefficients
	result=acf(data1)
	for i in range(1,10):
		feature.append(result[i])
#

	#pulse feature
	# print("时域特征提取")
	feature.append(np.mean(data2))
	feature.append(np.std(data2))
	feature.append(rms)
	ptp=np.max(data2)-np.min(data2)
	feature.append(ptp)
	feature.append(np.max(data2))
	feature.append(np.min(data2))
	feature.append(stats.skew(data2))
	feature.append(stats.kurtosis(data2))
	feature.append(np.median(data2))

	rms=math.sqrt(sum([x ** 2 for x in data2]) / len(data2))
	mean=np.mean(data2)
	amplitude=sum([abs(i-mean) for i in data2])/len(data2)
	feature.append(amplitude)
	diversion=sum([abs(i) for i in data2])/len(data2)
	feature.append(diversion)

	interval=[]
	for i in range(len(data2)-1):
		interval.append(data2[i+1]-data2[i])
	feature.append(np.max(interval))
	feature.append(np.min(interval))

	mean=np.mean(interval)
	amplitude=sum([abs(i-mean) for i in interval])/len(interval)
	feature.append(amplitude)

	diversion=sum([abs(i) for i in interval])/len(interval)
	feature.append(diversion)

	feature.append(stats.skew(interval))
	feature.append(stats.kurtosis(interval))
	feature.append(np.median(interval))


	# print("频域特征提取")

	#Frequency Domain
	fre=200
	freqs, data2fft=IAtool.fft(data2,fre)

	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(data2fft[i])
		else:
			break
	# print(extrafft)		
	feature.append(np.mean(extrafft))
	feature.append(np.std(extrafft))
	feature.append(np.max(extrafft)-np.min(extrafft))
	feature.append(np.max(extrafft))
	feature.append(np.min(extrafft))
	feature.append(np.median(extrafft))
	feature.append(stats.kurtosis(extrafft))
	feature.append(stats.skew(extrafft))

	# print("时频域特征提取")

	#Discrete Wavelet Transform
	coeffs = wavedec(data2, 'haar', level=3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))
	feature.append(np.std(cA3))
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	feature.append(rms)
	ptp=np.max(cA3)-np.min(cA3)
	feature.append(ptp)

	#Wigner Ville Distribution
	x,y,z=IAtool.WignerVillecal(data2)

	feature.append(np.max(z))
	feature.append(np.min(z))
	feature.append(np.max(z)-np.min(z))
	feature.append(np.std(z))
	maxindex=z.index(np.max(z))
	# maxindex=z.index(np.max(z))
	feature.append(x[maxindex])
	feature.append(y[maxindex])

	#Autoregressive Coefficients
	result=acf(data2)
	for i in range(1,10):
		feature.append(result[i])

	return feature