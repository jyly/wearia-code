# -*- coding=utf-8 -*-
import numpy as np
from pywt import wavedec
from scipy.stats import kurtosis,skew
import math
from normal_tool import *
import IAtool
from entropy.entropy import spectral_entropy,sample_entropy,perm_entropy
from EntroPy import multiscale_entropy


def motion_feature(data):
	data=meanfilt(data,10)
	data=np.array(data)
	datalen=len(data)
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
	rms=math.sqrt(sum([x ** 2 for x in data]) / datalen)
	diversion=sum([abs(i) for i in data])/datalen

	feature.append(rms)
	feature.append(diversion)

	sd1,sd2,ratio=poincare_plot(data)
	feature.append(sd1)
	feature.append(sd2)
	feature.append(ratio)


	#shannon entropy
	dataround=minmaxscale(data)
	dataround=[round(i,2) for i in dataround]
	dataround=np.array(dataround)
	dataroundreshape=dataround.reshape(-1,1)
	datashannonent=calcShannonEnt(dataroundreshape)
	# feature.append(datashannonent)

	#energy
	energy=0
	#contrast
	con=0
	# Inverse different moment
	IDM=0
	# Homogeneity
	hom=0
	for i in range(len(dataround)):
		energy=energy+dataround[i]*dataround[i]/datalen
		con=con+(1-i)*(1-i)*dataround[i]/datalen
		IDM=IDM+dataround[i]/(1+(1-i)*(1-i))
		hom=hom+dataround[i]/(1+abs(1-i))
	feature.append(energy)
	feature.append(con)	
	feature.append(IDM)
	feature.append(hom)

	# feature.append(spectral_entropy(dataround,sf=200))
	feature.append(sample_entropy(dataround,order=2))
	feature.append(perm_entropy(dataround, order=3, normalize=True))

	interval=[]
	for i in range(len(data)-1):
		interval.append(data[i+1]-data[i])

	feature.append(np.max(interval))
	feature.append(np.min(interval))
	feature.append(kurtosis(interval))
	feature.append(skew(interval))

	mean=np.mean(interval)
	rms=math.sqrt(sum([x ** 2 for x in interval]) / len(interval))
	diversion=sum([abs(i) for i in interval])/len(interval)
	
	feature.append(rms)
	feature.append(diversion)
 
	#inter shannon entropy
	interound=minmaxscale(interval)
	interound=[round(i,2) for i in interound]
	interound=np.array(interound)
	intereshape=interound.reshape(-1,1)
	intershannonent=calcShannonEnt(intereshape)
	feature.append(intershannonent)
	interlen=len(interound)
	#inter
	#energy
	energy=0
	#contrast
	con=0
	# Inverse different moment
	IDM=0
	# Homogeneity
	hom=0
	for i in range(len(interound)):
		energy=energy+interound[i]*interound[i]/interlen
		con=con+(1-i)*(1-i)*interound[i]/datalen
		IDM=IDM+interound[i]/(1+(1-i)*(1-i))
		hom=hom+interound[i]/(1+abs(1-i))
	feature.append(energy)
	feature.append(con)	
	feature.append(IDM)
	feature.append(hom)

	feature.append(sample_entropy(interound,order=2))
	feature.append(perm_entropy(interound, order=3, normalize=True))

	return feature



def ppg_feature(data):
	data=meanfilt(data,10)

	tempdata=IAtool.to2power(data)
	data=np.array(data)
	datalen=len(data)
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
	rms=math.sqrt(sum([x ** 2 for x in data]) / datalen)
	feature.append(rms)

	#Frequency Domain
	fre=200
	freqs, datafft=fft(tempdata,fre)
	extrafft=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extrafft.append(datafft[i])
		else:
			break
	feature.append(np.mean(extrafft))
	feature.append(np.std(extrafft))
	feature.append((np.max(extrafft)-np.min(extrafft)))
	feature.append(np.max(extrafft))
	feature.append(np.min(extrafft))
	feature.append(np.median(extrafft))
	feature.append(kurtosis(extrafft))
	feature.append(skew(extrafft))

	#Discrete Wavelet Transform
	coeffs = IAtool.dwt(tempdata,'haar',3)
	cA3, cD3, cD2 , cD1= coeffs
	feature.append(np.mean(cA3))	
	feature.append(np.std(cA3))
	feature.append((np.max(cA3)-np.min(cA3)))
	mean=np.mean(cA3)
	rms=math.sqrt(sum([x ** 2 for x in cA3]) / len(cA3))
	feature.append(rms)

	#Wigner Ville Distribution
	x,y,z=WignerVillecal(tempdata)
	feature.append(np.mean(z))
	feature.append(np.std(z))
	feature.append(np.max(z)-np.min(z))
	feature.append(np.max(z))
	feature.append(np.min(z))
	maxindex=z.index(np.max(z))
	feature.append(x[maxindex])
	feature.append(y[maxindex])

	#Autoregressive Coefficients
	for i in range(1,10):
		acf=get_auto_corr(data,i)
		feature.append(acf)

	#incre
	diversion=sum([abs(i) for i in data])/datalen
	feature.append(diversion)
 	diversion=sum([abs(i) for i in cA3])/len(cA3)
	feature.append(diversion)

	#庞加莱图
	sd1,sd2,ratio=poincare_plot(data)
	feature.append(sd1)
	feature.append(sd2)
	feature.append(ratio)
	#shannon entropy
	dataround=minmaxscale(data)
	dataround=[round(i,2) for i in dataround]
	dataround=np.array(dataround)
	dataroundreshape=dataround.reshape(-1,1)
	datashannonent=calcShannonEnt(dataroundreshape)
	feature.append(datashannonent)

	#energy
	energy=0
	#contrast
	con=0
	# Inverse different moment
	IDM=0
	# Homogeneity
	hom=0
	for i in range(len(dataround)):
		energy=energy+dataround[i]*dataround[i]/datalen
		con=con+(1-i)*(1-i)*dataround[i]/datalen
		IDM=IDM+dataround[i]/(1+(1-i)*(1-i))
		hom=hom+dataround[i]/(1+abs(1-i))
	feature.append(energy)
	feature.append(con)	
	feature.append(IDM)
	feature.append(hom)

	feature.append(sample_entropy(dataround,order=2))
	feature.append(perm_entropy(dataround, order=3, normalize=True))

	interval=[]
	for i in range(len(data)-1):
		interval.append(data[i+1]-data[i])

	feature.append(np.max(interval))
	feature.append(np.min(interval))
	feature.append(kurtosis(interval))
	feature.append(skew(interval))
	mean=np.mean(interval)
	rms=math.sqrt(sum([x ** 2 for x in interval]) / len(interval))
	diversion=sum([abs(i) for i in interval])/len(interval)
	feature.append(rms)
	feature.append(diversion)
 
	#inter shannon entropy
	interound=minmaxscale(interval)
	interound=[round(i,2) for i in interound]
	interound=np.array(interound)
	intereshape=interound.reshape(-1,1)
	intershannonent=calcShannonEnt(intereshape)
	feature.append(intershannonent)
	interlen=len(interound)
	
	#energy
	energy=0
	#contrast
	con=0
	# Inverse different moment
	IDM=0
	# Homogeneity
	hom=0
	for i in range(len(interound)):
		energy=energy+interound[i]*interound[i]/interlen
		con=con+(1-i)*(1-i)*interound[i]/datalen
		IDM=IDM+interound[i]/(1+(1-i)*(1-i))
		hom=hom+interound[i]/(1+abs(1-i))
	feature.append(energy)
	feature.append(con)	
	feature.append(IDM)
	feature.append(hom)
	feature.append(sample_entropy(interound,order=2))
	feature.append(perm_entropy(interound, order=3, normalize=True))

	#倒频谱
	fre=200
	freqs, caps=ceps(tempdata,fre)
	extracapst=[]
	for i in range(len(freqs)):
		if freqs[i]<5:
			extracapst.append(caps[i])
		else:
			break
	# print(len(extracapst))
	feature.append(np.mean(extracapst))
	feature.append(np.std(extracapst))
	feature.append((np.max(extracapst)-np.min(extracapst)))
	feature.append(np.max(extracapst))
	feature.append(np.min(extracapst))
	feature.append(np.median(extracapst))
	feature.append(kurtosis(extracapst))
	feature.append(skew(extracapst))

	# print("时频域特征提取")

	for i in range(len(extrafft)):
		feature.append(extrafft[i])
	for i in range(len(extracapst)):
		feature.append(extrafft[i])

	return feature

