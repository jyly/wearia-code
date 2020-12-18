# -*- coding=utf-8 -*-
import numpy as np
from siamese.siamese_combine_model import *
from classifier_tool import *
from normal_tool import *
import IAtool
import random
from itertools import combinations

def siamese_combine_class(datas,features,target,targetnum):
	feature=[]
	for i in range(len(target)):
		feature.append([])
		feature[i].append(datas[i])
		feature[i].append(features[i])


	tempfeature=IAtool.listtodic(feature,target)
	meanacc=[]
	meanfar=[]
	meanfrr=[]

	featurenum=30
	#选择的锚数
	anchornum=3
	#循环次数
	iternum=10
	#组合内序号个数
	testsetnumber=8
	#训练集个数
	# traincomnum=28

	rangek=list(range(0,targetnum-1))

	#得出用户数在comnum之间的组合
	# com=list(combinations(rangek,testsetnumber))
	# selectk = random.sample(com, iternum)	#在组合间，随机选其中的iternum个
	
	selectk=[]
	for t in range(iternum):
		selectk.append(random.sample(rangek, testsetnumber))
	
	for t in range(iternum):
		print("周期：",t)
		train_data=[]
		test_data=[]

		#用于限制训练集数量
		# selectks=[]
		# for i in rangek:
		# 	if i not in selectk[t]:
		# 		selectks.append(i)
		# selectks = random.sample(selectks, traincomnum)
		# print("被选择的训练集序号：",selectks)

		print("被选择的测试集序号：",selectk[t])
		for i in range(targetnum):
			if i in selectk[t]:
				test_data.append(tempfeature[i])
			# if i in selectks:
			else:
				train_data.append(tempfeature[i])	

		train_data,train_target,trainindex=IAtool.dictolist(train_data)
		test_data,test_target,testindex=IAtool.dictolist(test_data)
		
		print("训练集项目数：" ,trainindex)
		print("测试集项目数：",testindex)

		traindatas=[]
		trainfeatures=[]
		for i in range(len(train_data)):
			traindatas.append(train_data[i][0])
			trainfeatures.append(train_data[i][1])

		testdatas=[]
		testfeatures=[]
		for i in range(len(test_data)):
			testdatas.append(test_data[i][0])
			testfeatures.append(test_data[i][1])


		traindatas,testdatas,train_target,test_target=IAtool.datashape(traindatas,testdatas,train_target,test_target)
		#将n*m变为m*n
		# traindatas=IAtool.datatranspose(traindatas)
		# testdatas=IAtool.datatranspose(testdatas)
		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)

		
		trainfeatures,testfeatures,train_target,test_target=IAtool.datashape(trainfeatures,testfeatures,train_target,test_target)
		ppg_train_data,ppg_test_data,ppg_sort=IAtool.minepro(trainfeatures[:,0:76],testfeatures[:,0:76],train_target,featurenum)
		motion_train_data,motion_test_data,motion_sort=IAtool.minepro(trainfeatures[:,76:],testfeatures[:,76:],train_target,featurenum)
		trainfeatures=IAtool.datacombine(ppg_train_data,motion_train_data)
		testfeatures=IAtool.datacombine(ppg_test_data,motion_test_data)

		trainfeatures,testfeatures,scale_mean,scale_scale=IAtool.stdpro(trainfeatures,testfeatures)
		trainfeatures,testfeatures,train_target,test_target=IAtool.datashape(trainfeatures,testfeatures,train_target,test_target)

		train_data=[]
		for i in range(len(train_target)):
			train_data.append([])
			train_data[i].append(traindatas[i])
			train_data[i].append(trainfeatures[i])
		test_data=[]
		for i in range(len(test_target)):
			test_data.append([])
			test_data[i].append(testdatas[i])
			test_data[i].append(testfeatures[i])

		# score,label= siamese_weighted_combine(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
		score,label= siamese_mul_combine(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
		score=[i[0] for i in score]
		label=[i for i in label]
		print('原结果：',label)
		print('预测分数：',score)
		accuracy,far,frr=cal_siamese_eer(label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfar),")",)
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])
