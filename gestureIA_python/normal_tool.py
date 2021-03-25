# -*- coding=utf-8 -*-
import numpy as np
from scipy import signal,stats
import matplotlib.pyplot as plt
from minepy import MINE
from tftb.processing import WignerVilleDistribution,PseudoWignerVilleDistribution
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
import pywt
from math import log,sqrt
# from pyts.image import RecurrencePlot,MarkovTransitionField,GramianAngularField

#多个项目通用的工具

def indexpicshow(data):
	plt.plot(range(len(data)), data, 'blue')
	plt.show()

def pointpicshow(data):
	for i in range(len(data)):
		plt.plot(i, data[i], 'o')
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

def picshow(datax,datay):
	plt.plot(datax, datay, 'blue')
	plt.show()	

def mixmulpicshow(data1,data2,data3):
	plt.subplot(311)
	plt.plot(data3, data1, 'red')
	plt.subplot(312)
	plt.plot(data3, data2, 'blue')
	plt.subplot(313)
	plt.plot(data3, data1, 'red')
	plt.plot(data3, data2, 'blue')
	plt.show()

#快傅里叶变换，获取频域和对应的振幅
def fft(data, sampling_rate, fft_size=None):  
    if fft_size is None:  
        fft_size = len(data)  
    data = data[:fft_size]  
    #实数求解
    datafft = abs(np.fft.rfft(data)/fft_size*2)  
    freqs = np.linspace(0, int(1.0*sampling_rate/2), int(1.0*fft_size/2+1))    #linspace(0,100,501)
    return freqs, datafft    #频域，对应振幅


def ceps(data, sampling_rate, fft_size=None):  
    if fft_size is None:  
        fft_size = len(data)  
    data = data[:fft_size]  
    datafft = abs(np.fft.fft(data))  
    freqs = np.linspace(0, int(1.0*sampling_rate/2), int(1.0*fft_size))    #linspace(0,100,501)
    ceps=np.fft.ifft(np.log(np.abs(datafft))).real
    return freqs[1:], np.abs(ceps[1:])    #频域，对应振幅



def KL_divergence(P,Q):
	# KL = stats.entropy(P,Q)
	KL=sum(_p * log(_p / _q) for _p, _q in zip(P, Q) if _p != 0) 
	# KL=np.sum([v for v in P * np.log2(P/Q) if not np.isnan(v)])
	return KL

def JS_divergence(p,q):
	p=np.array(p)
	q=np.array(q)
	# Pnorm=sum([i for i in p])
	# Qnorm=sum([i for i in q])
	# _P = p / Pnorm
	# _Q = q / Qnorm
	# _P = p 
	# _Q = q 
	_P = p / np.linalg.norm(p, ord=1)
	_Q = q / np.linalg.norm(q, ord=1)
	M=(_P+_Q)/2
	JS=0.5*(KL_divergence(_P,M)+KL_divergence(_Q,M))
	# JS=0.5*stats.entropy(p, M)+0.5*stats.entropy(q, M)
	return JS

def cos_distance(A,B):
	cos = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
	return cos


def jaccard_distance(x,y):
	up=np.double(np.bitwise_and((x != y),np.bitwise_or(x != 0, y != 0)).sum())
	down=np.double(np.bitwise_or(x != 0, y != 0).sum())
	jac=(up/down)
	return jac

#计算相关系数
def calc_corr(a, b):
	a_avg = sum(a)/len(a)	
	b_avg = sum(b)/len(b) 	
	# 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n	
	cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)]) 	
	# 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
	sq = sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b])) 	
	corr_factor = cov_ab/sq 	
	return corr_factor

#3中标准化方法
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

def normalscale(data):
	scaler = preprocessing.Normalizer(norm='l1')
	data=scaler.fit_transform(np.array(data).reshape(-1,1))
	data=[i[0] for i in data]
	return data

#均值滤波
def meanfilt(datalist,interval,adddis=1):
	tempdata=[]
	#序列按间隔进行扩展
	for i in range(int(interval/2)):
		tempdata.append(datalist[0])
	for i in range(len(datalist)):
		tempdata.append(datalist[i])
	for i in range(int(interval/2)):
		tempdata.append(datalist[-1])
	filtlist=[]
	#对扩展后的数据进行滤波
	for i in range(0,len(datalist),adddis):
		filtlist.append(np.mean(tempdata[i:i+interval]))

	return filtlist

#3种butterworth滤波方法
def highpass(high,fre,data,order=3):#只保留高于high频率的信号，fre是采样频率,order是滤波器阶数
	wh=high/(fre/2)
	b, a = signal.butter(order, wh, 'high')
	data = signal.filtfilt(b, a, data)
	return data

def lowpass(low,fre,data,order=3):#只保留低于low频率的信号，fre是采样频率,order是滤波器阶数
	wl=low/(fre/2)
	b, a = signal.butter(order, wl, 'low')
	data = signal.filtfilt(b, a, data)
	return data

def bandpass(start,end,fre,data,order=3):#只保留start到end之间的频率的信号，fre是采样频率,order是滤波器阶数
	wa = start / (fre / 2) 
	we = end / (fre / 2) 
	b, a = signal.butter(order, [wa,we], 'bandpass')
	data = signal.filtfilt(b, a, data)
	return data


#计算香农熵
def calcShannonEnt(dataSet):
	dataSet=np.array(dataSet)
	total = len(dataSet)
	labelCounts = set(dataSet)
	eps = 1.4e-45
	shannonEnt = 0
	for key in labelCounts:
		prob = 1.0*len(np.where(dataSet==key)[0])
		shannonEnt = shannonEnt - (prob/total)*math.log(prob/total+eps,2)
	return shannonEnt 

#计算信息增益率
def NMI(A,B):
	# I(X,Y)=H(x)+H(Y)-H(XY)
	A=np.array(A)
	B=np.array(B)
	# len(A) should be equal to len(B)
	total = len(A)
	A_ids = set(A)
	B_ids = set(B)
	#Mutual information
	MI = 0
	eps = 1.4e-45
	for idA in A_ids:
		for idB in B_ids:
			idAOccur = np.where(A==idA)
			idBOccur = np.where(B==idB)
			idABOccur = np.intersect1d(idAOccur,idBOccur)
			px = 1.0*len(idAOccur[0])/total
			py = 1.0*len(idBOccur[0])/total
			pxy = 1.0*len(idABOccur)/total
			MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
	# Normalized Mutual information
	Hx = 0
	for idA in A_ids:
		idAOccurCount = 1.0*len(np.where(A==idA)[0])
		Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
	Hy = 0
	for idB in B_ids:
		idBOccurCount = 1.0*len(np.where(B==idB)[0])
		Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
	MIhat = (Hx+Hy-MI)
	MIhat = MIhat/Hy
	return MIhat


#计算多个特征的信息增益率
def minecal(data,target):
	informscore=[]
	for i in range(len(data[0])):
		tempdata=[]
		for j in range(len(data)):
			tempdata.append(data[j][i])
		# mine = MINE()
		# mine.compute_score(tempdata, target)
		# informscore.append(mine.mic())
		mine = NMI(tempdata, target)
		informscore.append(mine)
	informscore=np.array(informscore)
	informsort = np.argsort(-informscore)#由大到小排序，得到对应的序号
	temp=[i for i in informsort]
	informsort=temp
	print("informscore:",informscore)
	print("informsort:",informsort)
	return informsort

# 利用弹性网系数做特征提取
def elasticnet(data,target):
	regr = ElasticNet(random_state=0)
	regr.fit(data, target)
	print(regr.coef_)
	elasticscore=np.array(regr.coef_)
	elasticsort=np.argsort(-score)
	print("elasticscore:",elasticscore)
	print("elasticsort:",elasticsort)
	return elasticsort

# 维格纳分布
def WignerVillecal(data):	
	dist = PseudoWignerVilleDistribution(data)
	result = dist.run()
	tfr, times, freqs=result
	x=[]
	y=[]
	z=[]
	for i in range(len(times)):
		for j in range(len(freqs)):
			x.append(times[i])#时域序号点
			y.append(freqs[j])#频域序号点
			z.append(tfr[j][i])
	return x,y,z

#连续小波变换
def cwt(data,scales,wavelet='mexh'):
	coef, freqs=pywt.cwt(data,np.arange(1,scales),wavelet)
	return coef, freqs

#离散小波变换
def dwt(data,wavelet='haar',levels=3):
	coeffs = pywt.wavedec(data,wavelet,level=levels)
	return coeffs

# 第k阶的自相关系数
def get_auto_corr(timeSeries,k):
	l = len(timeSeries)
	#取出要计算的两个数组
	timeSeries1 = timeSeries[0:l-k]
	timeSeries2 = timeSeries[k:]
	timeSeries_mean = np.mean(timeSeries)
	# timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
	timeSeries_var = np.var(timeSeries)*l
	auto_corr = 0
	for i in range(l-k):
		temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
		auto_corr = auto_corr + temp 
	return auto_corr


# #序列改为递归图
# def recurrenceplot(data):
# 	result=np.array(data)
# 	rp = RecurrencePlot(threshold='point', percentage=20)
# 	X_rp = rp.fit_transform(result)
# 	# plt.imshow(X_rp[0], cmap='binary', origin='lower')
# 	# plt.title('Recurrence Plot', fontsize=16)
# 	# plt.tight_layout()
# 	# plt.show()
# 	# print(X_rp[0].shape)
# 	# print(X_rp[0])
# 	return X_rp


# #序列改为格拉姆角场图
# def gramianplot(data):
# 	result=np.array(data)
# 	gasf = GramianAngularField(image_size=32, method='summation')
# 	X_gasf = gasf.fit_transform(result)
# 	return X_gasf

#庞加莱图
def poincare_plot(data):
	x=[]
	y=[]
	for i in range(len(data)-1):
		x.append(data[i])	
		y.append(data[i+1])	
	cal=(2**0.5)/2
	xt=[]
	yt=[]
	for i in range(len(x)):
		xt.append(cal*(x[i]+y[i]))
		yt.append(cal*(-x[i]+y[i]))
	sd1=np.max(xt)-np.min(xt)
	sd2=np.max(yt)-np.min(yt)
	ratio=sd1/sd2
	return sd1,sd2,ratio