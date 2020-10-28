# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM,SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import EllipticEnvelope
from sklearn.externals import joblib
import IAtool
from classifier_tool import *
from normal_tool import *
import featurecontrol




def sklearn_TMC_mulclass(featureset,target,divnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	


	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)
		# train_data,test_data, train_target, test_target = train_test_split(featureset,target,train_size = 10,random_state = t*30,stratify=target)
		sampleslist=[]
		#获取单个类别的样本列
		print(train_target)
		for i in range(divnum):
			print("类别：",i)
			tempdata=[]
			for j in range(len(train_target)):
				if(train_target[j]==(i+1)):

					tempdata.append(train_data[j][0])
			samples=IAtool.calgestureprofile(tempdata)
			sampleslist.append(samples)

		finalfeatureset=[]
		for i in range(len(train_target)):
			tempfeature=[]
			temp=featurecontrol.ppg_feature(train_data[i][0])
			for j in temp:
				tempfeature.append(j)
			# temp=featurecontrol.ppg_feature(train_data[i][1])
			# for j in temp:
			# 	tempfeature.append(j)
			for j in range(len(sampleslist)):
				meandist=[]
				for k in range(3):
					d=IAtool.caldtw(sampleslist[j][k],train_data[i][0])
					print("i=",i,"j=",j,"k=",k,"d=",d)
					meandist.append(d)
				tempfeature.append(np.mean(meandist))
			finalfeatureset.append(tempfeature)
		train_data=finalfeatureset
		print(train_data)
		print(len(train_data))
		print(len(train_data[0]))

		finalfeatureset=[]
		for i in range(len(test_target)):
			tempfeature=[]
			temp=featurecontrol.ppg_feature(test_data[i][0])
			for j in temp:
				tempfeature.append(j)
			# temp=featurecontrol.ppg_feature(test_data[i][1])
			# for j in temp:
			# 	tempfeature.append(j)
			for j in range(len(sampleslist)):
				meandist=[]
				for k in range(3):
					d=IAtool.caldtw(sampleslist[j][k],test_data[i][0])
					print("i=",i,"j=",j,"k=",k,"d=",d)
					meandist.append(d)
				tempfeature.append(np.mean(meandist))
			finalfeatureset.append(tempfeature)
		test_data=finalfeatureset
		print(len(test_data))
		print(len(test_data[0]))
	
		print("进入第",t,"轮分类的信息熵降维阶段")
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)
		print("进入第",t,"轮分类的线性判别式分析阶段")
		train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)
	

		# print("进入第",t,"轮分类的主成分分析阶段")
		# train_data,test_data=IAtool.pcapro(trainfeatureset,trainfeatureset,8)

		print("进入第",t,"轮分类阶段")
		# clf = GradientBoostingClassifier(random_state=0)
		clf = SVC(probability=True)
		# clf = RandomForestClassifier()
		# clf = GaussianNB()
		# clf = MLPClassifier()
		# clf = KNeighborsClassifier(n_neighbors=13,weights='distance')

		clf.fit(X=train_data, y=train_target)


		result = clf.predict(test_data)
		score = clf.predict_proba(test_data)
		print('原结果：',test_target)
		print('预测结果：',result)
		print('预测分数：',score)

		# tp,tn,fp,fn=mul_accuracy_result(test_target,result,divnum)
	
		
		i=0
		while i<1:
			tp,tn,fp,fn=mul_accuracy_score(test_target,score,i,divnum)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			if frr>far:
				break
			i=i+0.001
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)

		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
	for i in range(len(meanacc)):
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])


def sklearn_mulclass(featureset,target,divnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)
		# train_data,test_data, train_target, test_target = train_test_split(featureset,target,train_size = 225,random_state = t*30,stratify=target)
		print("进入第",t,"轮分类的信息熵降维阶段")
		# train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)

		sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
		train_data=np.array(train_data)
		test_data=np.array(test_data)
		train_data=IAtool.scoreselect(train_data,sort,30)
		test_data=IAtool.scoreselect(test_data,sort,30)
		
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)
		print("进入第",t,"轮分类的线性判别式分析阶段")
		train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)
	

		# print("进入第",t,"轮分类的主成分分析阶段")
		# train_data,test_data=IAtool.pcapro(trainfeatureset,trainfeatureset,8)

		print("进入第",t,"轮分类阶段")
		# clf = GradientBoostingClassifier(random_state=0)
		clf = SVC(probability=True)
		# clf = RandomForestClassifier()
		# clf = GaussianNB()
		# clf = MLPClassifier()
		# clf = KNeighborsClassifier(n_neighbors=13,weights='distance')

		clf.fit(X=train_data, y=train_target)

		result = clf.predict(test_data)
		score = clf.predict_proba(test_data)
		print('原结果：',test_target)
		print('预测结果：',result)
		print('预测分数：',score)

		# tp,tn,fp,fn=mul_accuracy_result(test_target,result,divnum)
	
		i=0
		while i<1:
			tp,tn,fp,fn=mul_accuracy_score(test_target,score,i,divnum)
			accuracy=(tp+tn)/(tp+tn+fp+fn)
			far=(fp)/(fp+tn)
			frr=(fn)/(fn+tp)
			# print("i=",i)
			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
			if frr>far:
				break
			i=i+0.001
		print("i=",i)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)

		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
	for i in range(len(meanacc)):
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])


def sklearn_oneclass(featureset,target,classnum):#特征，目标，对第k个目标训练模型,从1开始
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)

		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)

		# print("进入第",t,"轮分类的线性判别式分析阶段")
		# train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)

		# print("进入第",t,"轮分类的主成分分析阶段")
		# oneclasstraindata,oneclasstestdata=IAtool.pcapro(oneclasstraindata,oneclasstestdata,8)

		print("进入第",t,"轮分类的oneclass阶段")

		oneclasstraindata=[]
		oneclasstestdata=test_data
		oneclasstesttarget=[]

		for k in range(len(train_target)):
			if train_target[k]==(classnum):
				oneclasstraindata.append(train_data[k]) 
		for k in range(len(test_target)):
			if test_target[k]==(classnum):
				oneclasstesttarget.append(1)
			else: 
				oneclasstesttarget.append(-1)
	
		clf = OneClassSVM(nu=0.02).fit(oneclasstraindata)
		# clf = EllipticEnvelope(random_state=0).fit(oneclasstraindata)
		# clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1).fit(oneclasstraindata)
		# clf = IsolationForest(random_state=0,max_features=len(oneclasstraindata[0]),bootstrap=True).fit(oneclasstraindata)

		joblib.dump(clf, 'model.pkl') 

		result = clf.predict(oneclasstestdata)
		scores = clf.score_samples(oneclasstestdata)
		# score = clf.predict_proba(featureset)
		dist=clf.decision_function(oneclasstestdata)
		print('原结果：',oneclasstesttarget)
		print('预测结果：',result)
		print('预测分数：',scores)
		print('模型距离：',dist)

		tp,tn,fp,fn=one_accuracy_score(oneclasstesttarget,dist,0)#result_scores: onesvm, 1. EllipticEnvelope,-80. IsolationForest,-0.64.LocalOutlierFactor -1
		# tp,tn,fp,fn=one_accuracy_result(oneclasstesttarget,result)
		print(tp,tn,fp,fn)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		print("accuracy:",accuracy,"far:",far,"frr:",frr)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))




def sklearn_finalmulclass(featureset,target,divnum):

	
	informsort,lda_bar,lda_scaling=IAtool.filterparameterread()
	lda_bar=np.array(lda_bar)
	lda_scaling=np.array(lda_scaling)
	featureset=IAtool.scoreselect(featureset,informsort,60)
	featureset=np.dot(featureset-lda_bar,lda_scaling)
	
	clf = joblib.load('model.pkl') 

	result = clf.predict(featureset)
	score = clf.predict_proba(featureset)
	dist = clf.decision_function(featureset)
	print('原结果：',target)
	print('预测结果：',result)
	print('预测分数：',score)
	print('模型距离：',dist)
	tp,tn,fp,fn=mul_accuracy_score(target,score,0.2,divnum)
	# tp,tn,fp,fn=mul_accuracy_result(target,result)

	print(tp,tn,fp,fn)
	accuracy=(tp+tn)/(tp+tn+fp+fn)
	far=(fp)/(fp+tn)
	frr=(fn)/(fn+tp)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)

def sklearn_finaloneclass(featureset,target,classnum):#特征，目标，对第k个目标训练模型,从1开始

	informsort,lda_bar,lda_scaling=IAtool.filterparameterread()
	lda_bar=np.array(lda_bar)
	lda_scaling=np.array(lda_scaling)
	
	featureset=IAtool.scoreselect(featureset,informsort,30)
	# featureset=np.dot(featureset-lda_bar,lda_scaling)

	finaltarget=[]
	for k in range(len(target)):
		if target[k]==(classnum):
			finaltarget.append(1)
		else: 
			finaltarget.append(-1)
	target=finaltarget

	clf = joblib.load('model.pkl') 

	result = clf.predict(featureset)
	score = clf.score_samples(featureset)
	dist = clf.decision_function(featureset)
	print('原结果：',target)
	print('预测结果：',result)
	print('预测分数：',score)
	print('模型距离：',dist)
	# tp,tn,fp,fn=one_accuracy_score(target,score,-1.5)
	tp,tn,fp,fn=one_accuracy_result(target,result)

	print(tp,tn,fp,fn)
	accuracy=(tp+tn)/(tp+tn+fp+fn)
	far=(fp)/(fp+tn)
	frr=(fn)/(fn+tp)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)



def sklearn_build_class(train_data,train_target,divnum):
	temp=[]
	train_data,temp,sort=IAtool.minepro(train_data,temp,train_target,30)


	# sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
	# train_data=np.array(train_data)
	# train_data=IAtool.scoreselect(train_data,sort,30)

	train_data,temp,lda_bar,lda_scaling=IAtool.ldapro(train_data,temp,train_target)
	IAtool.filterparameterwrite(sort,lda_bar,lda_scaling,'./parameter/stdpropara.txt')

	# print("进入第",t,"轮分类的主成分分析阶段")
	# train_data,test_data=IAtool.pcapro(trainfeatureset,trainfeatureset,8)

	# clf = GradientBoostingClassifier(random_state=0)
	clf = SVC(probability=True)
	# clf = RandomForestClassifier()
	# clf = GaussianNB()
	# clf = MLPClassifier()
	# clf = KNeighborsClassifier(n_neighbors=13,weights='distance')
	clf.fit(X=train_data, y=train_target)
	joblib.dump(clf, './parameter/model.pkl') 
	result = clf.predict(train_data)
	score = clf.predict_proba(train_data)
	print('原结果：',train_target)
	print('预测结果：',result)
	print('预测分数：',score)
	# tp,tn,fp,fn=mul_accuracy_result(test_target,result,divnum)
	i=0
	while i<1:
		tp,tn,fp,fn=mul_accuracy_score(train_target,score,i,divnum)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		# print("i=",i)
		# print("accuracy:",accuracy,"far:",far,"frr:",frr)
		if frr>far:
			break
		i=i+0.01
	print("i=",i)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)


def sklearn_final_class(test_data,test_target,divnum):
	informsort,lda_bar,lda_scaling=IAtool.filterparameterread('./parameter/stdpropara.txt')
	lda_bar=np.array(lda_bar)
	lda_scaling=np.array(lda_scaling)
	
	test_data=IAtool.scoreselect(test_data,informsort,30)
	test_data=np.dot(test_data-lda_bar,lda_scaling)

	clf = joblib.load('./parameter/model.pkl') 

	result = clf.predict(test_data)
	# score = clf.score_samples(test_data)
	# dist = clf.decision_function(test_data)
	dist = clf.predict_proba(test_data)
	print('原结果：',test_target)
	print('预测结果：',result)
	# print('预测分数：',score)
	print('模型距离：',dist)
	# tp,tn,fp,fn=one_accuracy_score(target,score,-1.5)
	# tp,tn,fp,fn=mul_accuracy_result(test_target,result,divnum)
	tp,tn,fp,fn=mul_accuracy_score(test_target,dist,0.5,divnum)
	print(tp,tn,fp,fn)
	accuracy=(tp+tn)/(tp+tn+fp+fn)
	far=(fp)/(fp+tn)
	frr=(fn)/(fn+tp)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)