# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from siamese.siamese_model import *
from classifier_tool import *
from normal_tool import *
import IAtool
import random
import filecontrol 
from itertools import combinations

#对原始序列输入到孪生网络中
def siamese_oridata_classifier(dataset,target,targetnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	dataset=np.array(dataset)
	target=np.array(target)

	for t in range(0,1):

		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)

		# train_data=np.array(train_data)
		# train_target=np.array(train_target)
		# test_data=np.array(test_data)
		# test_target=np.array(test_target)
	

		#将2*300变为300*2
		# temptraindata=[]
		# for i in range(len(train_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(train_data[0])):
		# 			temp[j].append(test_data[i][k][j])
		# 	temptraindata.append(temp)
		# temptestdata=[]
		# for i in range(len(test_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(train_data[0]):
		# 			temp[j].append(test_data[i][k][j])
		# 	temptestdata.append(temp)
		# train_data=temptraindata
		# test_data=temptestdata


		#自己的方案
		test_score,test_label= siamese_oridata(train_data,test_data, train_target, test_target,targetnum)
		#emgauth方案
		# score,test_label,threshold= siamese_emgauth(train_data,test_data, train_target, test_target,targetnum)
		# print(score)
		
		test_score=[i[0] for i in test_score]
		print('原结果：',test_label)
		print('预测分数：',test_score)
		i=0.01
		while i<5:
			tp,tn,fp,fn=siamese_accuracy_score(test_label,test_score,i)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			if (frr-far)<0.02:
				break
			i=i+0.005
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))



def siamese_cwt_classifier(dataset,target,targetnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,1):
		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)

		print("进入cwt处理阶段")	
		temptraindata=[]
		temptestdata=[]
		#提取1-25频段的时频图
		for i in range(len(train_data)):
			temp=[]
			for j in range(len(train_data[0])):
				coef, freqs=cwt(train_data[i][j],25,'mexh')
				temp.append(coef)
			temptraindata.append(temp)

		for i in range(len(test_data)):
			temp=[]
			for j in range(len(test_data[0])):
				coef, freqs=cwt(test_data[i][j],25,'mexh')
				temp.append(coef)
			temptestdata.append(temp)
		print("cwt处理完成")	
		train_data=temptraindata
		test_data=temptestdata


		# print("数据重排序(方案1)")	2*24*300到24*300*2
		temptraindata=[]
		temptestdata=[]
		for i in range(len(train_data)):
			temp=[]
			for j in range(24):
				temp.append([])
				for k in range(300):
					temp[j].append([])
					for d in range(len(train_data[0])):
						temp[j][k].append(train_data[i][d][j][k])
			temptraindata.append(temp)
		for i in range(len(test_data)):
			temp=[]
			for j in range(24):
				temp.append([])
				for k in range(300):
					temp[j].append([])
					for d in range(len(test_data[0])):
						temp[j][k].append(test_data[i][d][j][k])
			temptestdata.append(temp)
		train_data=temptraindata
		test_data=temptestdata

		# print("数据重排序(方案2)")	2*24*300到300*24*2

		# temptraindata=[]
		# temptestdata=[]
		# for i in range(len(train_data)):
		# 	temp=[]
		# 	for d in range(len(train_data[0])):
		# 		temp.append([])
		# 		for j in range(300):
		# 			temp[d].append([])
		# 			for k in range(24):
		# 				temp[d][j].append(train_data[i][d][k][j])
		# 	temptraindata.append(temp)
		# for i in range(len(test_data)):
		# 	temp=[]
		# 	for d in range(len(test_data[0])):
		# 		temp.append([])
		# 		for j in range(300):
		# 			temp[d].append([])
		# 			for k in range(24):
		# 				temp[d][j].append(test_data[i][d][k][j])
		# 	temptestdata.append(temp)
		# train_data=temptraindata
		# test_data=temptestdata


		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)

		# train_data=train_data.astype('float32')
		# test_data=test_data.astype('float32')

		#自己的方案
		score,test_label= siamese_cwt(train_data,test_data, train_target, test_target,targetnum)
		#cwt_emg方案
		# score,test_label,threshold= siamese_cwt_emg(train_data,test_data, train_target, test_target,targetnum)
		print(score)
		score=[i[0] for i in score]
		print('原结果：',test_label)
		print('预测分数：',score)
		i=0.1
		while i<5:
			tp,tn,fp,fn=siamese_accuracy_score(test_label,score,i)
			print(tp,tn,fp,fn)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			i=i+0.05
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))

def siamese_feature_classifier(dataset,target,targetnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,5):
		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)


		print("进入第",t,"轮分类的信息熵降维阶段")
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)

		# print("进入第",t,"轮分类的线性判别式分析阶段")
		# train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)
		# IAtool.filterparameterwrite(sort,lda_bar,lda_scaling,'./ldapropara.txt')
		
		# train_data,test_data,pca_mean,pca_components=IAtool.pcapro(train_data,test_data)
		# IAtool.filterparameterwrite(sort,pca_mean,pca_components,'./pcapropara.txt')

		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)
		IAtool.filterparameterwrite(sort,scale_mean,scale_scale,'./stdpropara.txt')

		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)
		print("训练集样本数：",train_data.shape)
		print("测试集样本数：",test_data.shape)

		# train_data=train_data.astype('float32')
		# test_data=test_data.astype('float32')

		score,test_label= siamese_feature(train_data,test_data, train_target, test_target,targetnum)
		print('原结果：',test_label)
		score=[i[0] for i in score]
		print('预测分数：',score)

		i=0.001
		while i<3:
			tp,tn,fp,fn=siamese_accuracy_score(test_label,score,i)
			# print(tp,tn,fp,fn)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			if (frr-far)<0.02:
				break

			i=i+0.001
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))



def siamese_ori_final_class(dataset,target,targetnum):
	dataset=np.array(dataset)
	target=np.array(target)
	score,label= siamese_ori_final(dataset,target,targetnum)
	print(score)
	
	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	i=0.1
	while i<3:
		tp,tn,fp,fn=siamese_accuracy_score(label,score,i)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		if abs(far-frr)<0.02:
			break
		i=i+0.001
	print("i=",i)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)



def siamese_feature_final_class(dataset,target,targetnum):


	dataset=np.array(dataset)
	target=np.array(target)

	# sort,lda_bar,lda_scaling=IAtool.filterparameterread('./ldapropara.txt')
	# lda_bar=np.array(lda_bar)
	# lda_scaling=np.array(lda_scaling)

	sort,pca_mean,pca_components=IAtool.filterparameterread('./pcapropara.txt')
	pca_mean=np.array(pca_mean)
	pca_components=np.array(pca_components)

	sort,scale_mean,scale_scale=IAtool.filterparameterread('./stdpropara.txt')
	scale_mean=np.array(scale_mean)
	scale_scale=np.array(scale_scale)

	dataset=IAtool.scoreselect(dataset,sort,30)
	# dataset=np.dot(dataset-lda_bar,lda_scaling)
	dataset=np.dot(dataset-pca_mean, pca_components.T)
	dataset=(dataset-scale_mean)/scale_scale

	score,label= siamese_feature_final(dataset,target,targetnum)

	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	i=0.1
	while i<5:
		tp,tn,fp,fn=siamese_accuracy_score(label,score,i)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		if abs(far-frr)<0.02:
			break
		i=i+0.005
	print("i=",i)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)

def siamese_feature_divide_class(feature,target,targetnum):

	#将特征分为不同的人和手势类
	tempfeature=[]
	maxusernum=int(targetnum/9)
	print("数据集中的用户数：",maxusernum)
	for i in range(maxusernum):
		tempfeature.append([])
		for j in range(9):
			tempfeature[i].append([])
	for i in range(len(target)):
		t1=int((target[i]-1)/9)
		t2=(target[i]-1)%9
		tempfeature[t1][t2].append(feature[i])
	meanacc=[]
	meanfar=[]
	meanfrr=[]

	#循环次数
	iternum=20
	#组合内序号个数
	comnum=5

	rangek=list(range(0,maxusernum))
	#得出用户数在2之间的组合
	com=list(combinations(rangek,comnum))
	#在组合间，随机选其中的10个
	selectk = random.sample(com, iternum)
	for t in range(iternum):
		print("周期：",t)
		train_data=[]
		test_data=[]

		print("被选择的测试集序号：",selectk[t])
		for i in range(maxusernum):
			if i in selectk[t]:
				test_data.append(tempfeature[i])
			# if i in selectks:
			else:
				train_data.append(tempfeature[i])	

		temptraindata=[]
		temptraintarget=[]
		trainindex=1
		for i in range(len(train_data)):
			for j in range(0,1):
				for k in range(len(train_data[i][j])):
					temptraindata.append(train_data[i][j][k])
					temptraintarget.append(trainindex)
				trainindex=trainindex+1
		trainindex=trainindex-1
		print("训练集项目数：" ,trainindex)

		temptestdata=[]
		temptesttarget=[]
		testindex=1
		for i in range(len(test_data)):
			for j in range(0,1):
				for k in range(len(test_data[i][j])):
					temptestdata.append(test_data[i][j][k])
					temptesttarget.append(testindex)
				testindex=testindex+1
		testindex=testindex-1
		print("测试集项目数：",testindex)

		train_data=temptraindata
		train_target=temptraintarget
		test_data=temptestdata
		test_target=temptesttarget
		#将所需的不同手势类别划分为训练集和测试集


		featurenum=30
		temp=[]
		train_data,temp,sort=IAtool.minepro(train_data,temp,train_target,featurenum)

		# train_data,temp,lda_bar,lda_scaling=IAtool.ldapro(train_data,temp,train_target)
		# IAtool.filterparameterwrite(sort,lda_bar,lda_scaling,'./ldapropara.txt')
		
		# train_data,temp,pca_mean,pca_components=IAtool.pcapro(train_data,temp)
		# IAtool.filterparameterwrite(sort,pca_mean,pca_components,'./pcapropara.txt')

		train_data,temp,scale_mean,scale_scale=IAtool.stdpro(train_data,temp)
		IAtool.filterparameterwrite(sort,scale_mean,scale_scale,'./stdpropara.txt')


		train_data=np.array(train_data)
		train_target=np.array(train_target)
		print("train_data.shape:",train_data.shape)
		siamese_feature_buildmodel(train_data,train_target,trainindex)
		#以上利用所有训练集数据建立模型

		test_data=np.array(test_data)
		test_target=np.array(test_target)

		# sort,lda_bar,lda_scaling=IAtool.filterparameterread('./ldapropara.txt')
		# lda_bar=np.array(lda_bar)
		# lda_scaling=np.array(lda_scaling)

		# sort,pca_mean,pca_components=IAtool.filterparameterread('./pcapropara.txt')
		# pca_mean=np.array(pca_mean)
		# pca_components=np.array(pca_components)

		sort,scale_mean,scale_scale=IAtool.filterparameterread('./stdpropara.txt')
		scale_mean=np.array(scale_mean)
		scale_scale=np.array(scale_scale)

		test_data=IAtool.scoreselect(test_data,sort,featurenum)
		# test_data=np.dot(test_data-lda_bar,lda_scaling)
		# test_data=np.dot(test_data-pca_mean, pca_components.T)
		test_data=(test_data-scale_mean)/scale_scale


		score,label= siamese_feature_final(test_data,test_target,testindex)
	
		score=[i[0] for i in score]
		print('原结果：',label)
		print('预测分数：',score)
		i=0.001
		while i<3:
			tp,tn,fp,fn=siamese_accuracy_score(label,score,i)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			if (frr-far)<0.02:
				break
			i=i+0.001
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])
