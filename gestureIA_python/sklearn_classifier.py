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

def sklearn_mulclass(featureset,target,divnum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)
		# train_data,test_data, train_target, test_target = train_test_split(featureset,target,train_size = 690,random_state = t*30,stratify=target)
		print("进入第",t,"轮分类的信息熵降维阶段")
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,30)
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)
		print("进入第",t,"轮分类的线性判别式分析阶段")
		train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)
	
		IAtool.filterparameterwrite(sort,lda_bar,lda_scaling)

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

		joblib.dump(clf, 'model.pkl') 

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