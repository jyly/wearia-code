# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from tripletloss.tripletloss_model import *
from classifier_tool import *
from normal_tool import *
import IAtool
from itertools import combinations


def tripletloss_oridata_classifier(dataset,target,targetnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	dataset=np.array(dataset)
	target=np.array(target)
	for t in range(0,1):
		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)

		score,test_label= tripletloss_ori(train_data,test_data, train_target, test_target,targetnum)
		print('原结果：',test_label)
		print('预测分数：',score)
		# score=[i[0] for i in score]

		k=1
		while k<100:
			tp,tn,fp,fn=tripletloss_accuracy_score(test_label,score,k)
			print(tp,tn,fp,fn)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			print("k=",k)
			print("accuracy:",accuracy,"far:",far,"frr:",frr)
			k=k+0.5

def tripletloss_feature_classifier(dataset,target,targetnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	dataset=np.array(dataset)
	target=np.array(target)
	for t in range(0,1):
		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)

		print("进入第",t,"轮分类的信息熵降维阶段")
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)
		# print("进入第",t,"轮分类的线性判别式分析阶段")
		# train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)

		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)
		IAtool.filterparameterwrite(sort,scale_mean,scale_scale,'./stdpropara.txt')

		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)
		print(train_data.shape)
		print(test_data.shape)


		score,test_label= tripletloss_feature(train_data,test_data, train_target, test_target,targetnum)
		print('原结果：',test_label)
		print('预测分数：',score)
		# score=[i[0] for i in score]

		i=0.01
		while i<10:
			tp,tn,fp,fn=tripletloss_accuracy_score(test_label,score,i)
			# print(tp,tn,fp,fn)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			if (frr-far)<0.02:
				break
			i=i+0.01
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))




def tripletloss_feature_build_class(train_data,train_target,trainindex,featurenum):

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
	tripletloss_feature_buildmodel(train_data,train_target,trainindex)





def tripletloss_feature_final_class(test_data,test_target,targetnum,featurenum,anchornum):

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

	score,label= tripletloss_feature_final(test_data,test_target,targetnum,anchornum)
	# score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	i=0.01
	while i<3:
		tp,tn,fp,fn=tripletloss_accuracy_score(label,score,i)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		# print("i=",i)
		# print("accuracy:",accuracy,"far:",far,"frr:",frr)
		if frr<far:
			break
		i=i+0.01
	print("i=",i)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)
	return accuracy,far,frr




def tripletloss_feature_divide_classifier(feature,target,targetnum):

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
	iternum=30
	#组合内序号个数
	comnum=8

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
		anchornum=18

		tripletloss_feature_build_class(train_data,train_target,trainindex,featurenum)
		accuracy,far,frr=tripletloss_feature_final_class(test_data,test_target,testindex,featurenum,anchornum)

		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])

		