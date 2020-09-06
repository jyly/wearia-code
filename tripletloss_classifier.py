# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from tripletloss.tripletloss_model import *
from classifier_tool import *
from normal_tool import *
import IAtool
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
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,60)
		# print("进入第",t,"轮分类的弹性网降维阶段")
		# train_data,test_data=IAtool.elasticnetpro(train_data,test_data,train_target,30)
		print("进入第",t,"轮分类的线性判别式分析阶段")
		train_data,test_data,lda_bar,lda_scaling=IAtool.ldapro(train_data,test_data,train_target)


		score,test_label= tripletloss_feature(train_data,test_data, train_target, test_target,targetnum)
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