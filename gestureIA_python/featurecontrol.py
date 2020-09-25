# -*- coding=utf-8 -*-
import numpy as np
from pywt import wavedec
from scipy.stats import kurtosis,skew
import math
from normal_tool import *
import IAtool
#特征原集
def ppg_feature(data):
	tempdata=IAtool.to2power(data)
	data=np.array(data)
	tempdata=np.array(tempdata)
	feature=[]
	# print("时域特征提取")
	feature.append(np.mean(data))
	feature.append(np.std(data))
	feature.append((np.max(data)-np.min(data)))
	feature.append(np.max(data))
	feature.append(np.min(data))
	feature.append(np.median(data))
	feature.append(kurtosis(data))
	feature.append(skew(data))


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
	feature.append(kurtosis(interval))
	feature.append(skew(interval))
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
	freqs, datafft=fft(tempdata,fre)

	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(datafft[i])
		else:
			break
	# print(extrafft)		
	feature.append(np.mean(extrafft))
	feature.append(np.std(extrafft))
	feature.append((np.max(extrafft)-np.min(extrafft)))
	feature.append(np.max(extrafft))
	feature.append(np.min(extrafft))
	feature.append(np.median(extrafft))
	feature.append(kurtosis(extrafft))
	feature.append(skew(extrafft))

	# print("时频域特征提取")

	#Discrete Wavelet Transform
	coeffs = IAtool.dwt(tempdata,'haar',3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))	
	feature.append(np.std(cA3))
	feature.append((np.max(cA3)-np.min(cA3)))


	mean=np.mean(cA3)
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	amplitude=sum([abs(i-mean) for i in cA3])/len(cA3)
	diversion=sum([abs(i) for i in cA3])/len(cA3)
	
	feature.append(rms)
	feature.append(amplitude)
	feature.append(diversion)

	#Wigner Ville Distribution
	# x,y,z=WignerVillecal(tempdata)
	# feature.append(np.mean(z))
	# feature.append(np.std(z))
	# feature.append(np.max(z)-np.min(z))
	# feature.append(np.max(z))
	# feature.append(np.min(z))
	# maxindex=z.index(np.max(z))
	# feature.append(x[maxindex])
	# feature.append(y[maxindex])

	#Autoregressive Coefficients
	for i in range(1,10):
		acf=get_auto_corr(data,i)
		feature.append(acf)
	return feature

def motion_feature(data):
	feature=[]
	# print("时域特征提取")
	feature.append(np.mean(data))
	feature.append(np.std(data))
	feature.append((np.max(data)-np.min(data)))
	feature.append(np.max(data))
	feature.append(np.min(data))
	feature.append(np.median(data))
	feature.append(kurtosis(data))
	feature.append(skew(data))


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
	feature.append(kurtosis(interval))
	feature.append(skew(interval))
	feature.append(np.median(interval))

	mean=np.mean(interval)
	rms=math.sqrt(sum([x ** 2 for x in interval]) / len(interval))
	amplitude=sum([abs(i-mean) for i in interval])/len(interval)
	diversion=sum([abs(i) for i in interval])/len(interval)
	
	feature.append(rms)
	feature.append(amplitude)
	feature.append(diversion)
	return feature

