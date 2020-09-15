# -*- coding=utf-8 -*-
import numpy as np
import IAtool
import os
from scipy import stats
import matplotlib.pyplot as plt
import math
import featureextra

def energycal(data,win,addwin=1,threshold=0.2):
	energy=[]
	for i in range(0,len(data)-win,addwin):
		tempenergy=0
		for j in range(i,i+win,addwin):
			tempenergy=tempenergy+(data[j]-threshold)*data[j]
		energy.append(tempenergy)
	return energy



def MAserach(dn,fre):
	oristd=[]
	winslen=200
	for i in range(len(dn)-winslen):
		ori=np.std(dn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)


	pointstartindex=0
	pointendindex=0
	tag=0
	i=len(oristd)
	lens=int(0.8*fre)
	while i >2*lens:
		i=i-1
		# print(i)
		#从后往前判断，当大于阈值时，认为可能存在手势
		if oristd[i]>1:

			end=i
			flag=0
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(i-lens,i):
				if oristd[j]<1:
					flag=1
					break
			
			if 0==flag:
				start=i-3*lens
				if start<0:
					start=0
				for j in range(start,i-lens):
					pointstartindex=j
					if oristd[j]>1:
						break
				pointstartindex=pointstartindex+int(0.5*fre)
				pointendindex=i+int(0.5*fre)
				tag=1
				break

	return tag,pointstartindex,pointendindex


def to2power(data):
	if len(data)<256:
		for i in range((256-len(data))):
			data.append(0)
	return data	


# filepath='./2020-05-01-17-22-00.csv'#静止状态1
# filepath='./2020-05-01-17-37-04.csv'#静止状态2
# filepath='./2020-05-01-17-37-24.csv'#静止状态3
# filepath='./2020-05-01-17-38-03.csv'#静止状态4
# filepath='./2020-05-03-14-13-52.csv'#静止状态5


# filepath='./2020-05-01-16-47-43.csv'#手势1
# filepath='./2020-05-01-16-53-55.csv'#手势2
# filepath='./2020-05-01-16-44-56.csv'#手势3
# filepath='./2020-05-01-17-41-48.csv'#手势4
# filepath='./2020-05-01-17-42-30.csv'#手势5


# filepath='./2020-05-01-17-46-17.csv'#走路摆臂
# filepath='./2020-05-01-17-49-12.csv'#打字
# filepath='./2020-05-01-17-50-45.csv'#扩胸
# filepath='./2020-05-01-17-54-20.csv'#拿起东西

# filepath='2020-05-04-08-59-02.csv'
# filepath='2020-05-04-12-06-53.csv'
# filepath='2020-05-04-12-11-20.csv'



# filepath='ica-2020-05-04-08-59-02.csv'#ica 静止1
# filepath='ica-2020-05-04-12-06-53.csv'#ica 手势3
# filepath='ica-2020-05-04-12-11-20.csv'#ica 手势3
# filepath='./data/left/1/2020-04-06-18-50-50.csv'	#遍历不同用户同一手势文件
filepath='./icappg.csv'
# filepath='./temp.txt'

def calc_corr(a, b):
	a_avg = sum(a)/len(a)	
	b_avg = sum(b)/len(b) 	
	# 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n	
	cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)]) 	
	# 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
	sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b])) 	
	print(cov_ab,sq)
	corr_factor = cov_ab/sq 	
	return corr_factor


# temp=[]
# input_1=open(filepath,'r+')
# for row in input_1:
# 	temp.append(float(row.replace('\n','')))
# input_1.close()
# print(temp)
# IAtool.indexpicshow(temp)


ppgx=[]
ppgy=[]
timestamp=[]
input_1=open(filepath,'r+')
for row in input_1:
	row=list(eval(row))
	ppgx.append(row[0])
	ppgy.append(row[1])
	# timestamp.append(row[2])
input_1.close()

IAtool.mixindexpicshow(ppgx,ppgy)

print(calc_corr(ppgx,ppgy))

# tag,pointstartindex,pointendindex=MAserach(ppgx,200)

# if 0==tag:
# 	print("当前片段不存在手势")
# else:
# 	print(pointstartindex,pointendindex)
# 	plt.plot(range(len(ppgx)), ppgx, 'red')
# 	plt.axvline(pointstartindex)
# 	plt.axvline(pointendindex)
# 	plt.show()
# data=ppgx[pointstartindex:pointendindex]
# # data=to2power(data)
# # print(len(data))
# # data=np.array(data)
# feature=featureextra.orifeatureextra(data)
# for i in range(0,len(feature)):
# 	print(i)
# 	print(feature[i])



# print(data)
# fre=200
# freqs, datafft=IAtool.fft(data,fre)
# print(freqs)
# print(datafft)
# data=to2power(data)
# from pywt import wavedec
# coeffs = wavedec(data, 'haar', level=3)
# print(coeffs[0])




# cA4, cD4, cD3, cD2 , cD1= coeffs
# cA3, cD3, cD2 , cD1= coeffs
# cA2,  cD2 , cD1= coeffs
# print(cA3)
# print(cD3)
# print(cA2)
# print(cD2)
# print(cD1)
# plt.subplot(311)
# plt.plot(data)

# plt.subplot(312)
# plt.plot(cA4)

# plt.subplot(313)
# plt.plot(cD4)

# plt.show()

'''

'''
# print(np.mean(ppgx),np.mean(ppgy))
# print(np.std(ppgx),np.std(ppgy))
# print(stats.kurtosis(ppgx),stats.kurtosis(ppgy))
# print(stats.skew(ppgx),stats.skew(ppgy))

# xenergy=energycal(ppgx,250)
# yenergy=energycal(ppgy,250)

# print()
# print(np.mean(xenergy),np.mean(yenergy))
# print(np.std(xenergy),np.std(yenergy))
# print(stats.kurtosis(xenergy),stats.kurtosis(yenergy))
# print(stats.skew(xenergy),stats.skew(yenergy))

# filepath='./raw-2020-05-01-12-06-29.csv'

# ppgx=[]
# ppgy=[]
# timestamp=[]
# input_1=open(filepath,'r+')
# for row in input_1:
# 	row=list(eval(row))
# 	ppgx.append(row[0])
# 	ppgy.append(row[1])
# 	# timestamp.append(row[2])
# input_1.close()

# IAtool.mixindexpicshow(ppgx,ppgy)


# from libsvm import svmutil
# from libsvm import svm

#y 划分类，x中1：和2：分别对应第一个和第二个特征

# y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
# x=[]
# temp={}
# temp[1]=1
# temp[2]=1
# x.append(temp)
# temp={}
# temp[1]=-1
# temp[2]=-1
# x.append(temp)
# print(x)


# prob = svmutil.svm_problem(y, x)
# param = svmutil.svm_parameter('-t 0 -c 4 -b 1')
# # model = svmutil.svm_train(prob, param)
# model=svmutil.svm_load_model("test.model")

# yt = [1]
# xt = [{1: 1, 2: 1}]
# p_label, p_acc, p_val = svmutil.svm_predict(yt, xt, model)
# # svmutil.svm_save_model("test.model",model)
# print(p_label)
# print(p_acc)
# print(p_val)



# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2,2])

# clf = LinearDiscriminantAnalysis()
# clf.fit(X, y)
# print(clf.coef_)
# print(clf.means_)
# print(clf.intercept_)
# print(clf.explained_variance_ratio_)
# print(clf.priors_)
# print(clf.scalings_)
# print(clf.covariance_)

# print(clf.xbar_)
# print(clf.classes_)
# X1=clf.transform(X)
# X2= np.dot(X, clf.scalings_)
# print("X1")
# print(X1)
# print("X2")
# print(X2)


