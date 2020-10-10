# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from siamese.siamese_data_model import *
from siamese.siamese_base import *
from classifier_tool import *
from normal_tool import *
import IAtool
import random
import filecontrol 
from itertools import combinations



# 对原始序列输入到孪生网络中
def siamese_data_based_classifier(dataset,target,targetnum,anchornum):
	meanacc=[]
	meanfar=[]
	meanfrr=[]	
	dataset=np.array(dataset)
	target=np.array(target)
	for t in range(0,10):
		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)	

		# 将n*300变为300*n
		# temptraindata=[]
		# for i in range(len(train_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(train_data[0])):
		# 			temp[j].append(train_data[i][k][j])
		# 	temptraindata.append(temp)
		# train_data=temptraindata
		# temptestdata=[]
		# for i in range(len(test_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(test_data[0])):
		# 			temp[j].append(test_data[i][k][j])
		# 	temptestdata.append(temp)
		# test_data=temptestdata

		#自己的方案
		score,test_label= siamese_data(train_data,test_data, train_target, test_target,targetnum,targetnum,anchornum)
		score=[i[0] for i in score]
		print('原结果：',test_label)
		print('预测分数：',score)
		accuracy,far,frr=cal_siamese_eer(test_label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfar),")",)
	for i in range(len(meanacc)):
		# print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])


def siamese_data_build_class(train_data,train_target,trainindex):
	#将2*300转为300*2
	# temptraindata=[]
	# for i in range(len(train_data)):
	# 	temp=[]
	# 	for j in range(300):
	# 		temp.append([])
	# 		for k in range(len(train_data[0])):
	# 			temp[j].append(train_data[i][k][j])
	# 	temptraindata.append(temp)
	# train_data=temptraindata
	siamese_data_buildmodel(train_data,train_target,trainindex)

def siamese_data_final_class(test_data,test_target,targetnum,anchornum):
	# temptestdata=[]
	# for i in range(len(test_data)):
	# 	temp=[]
	# 	for j in range(300):
	# 		temp.append([])
	# 		for k in range(len(test_data[0])):
	# 			temp[j].append(test_data[i][k][j])
	# 	temptestdata.append(temp)
	# test_data=temptestdata
	score,label= siamese_data_final(test_data,test_target,targetnum,anchornum)
	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	accuracy,far,frr=cal_siamese_eer(label,score)
	return accuracy,far,frr


def siamese_data_divide_class(feature,target,targetnum):
	#类别-样本-多个特征
	# 只有一个手势的情况
	maxusernum=targetnum
	print("数据集中的用户数：",maxusernum)
	tempfeature=[[] for i in range(maxusernum)]
	for i in range(len(target)):
		tempfeature[int((target[i]-1))].append(feature[i])

	meanacc=[]
	meanfar=[]
	meanfrr=[]

	#循环次数
	iternum=5
	#组合内序号个数
	comnum=40
	#训练集个数
	# traincomnum=28

	rangek=list(range(0,targetnum-1))
	#得出用户数在2之间的组合
	# com=list(combinations(rangek,comnum))
	#在组合间，随机选其中的iternum个
	# selectk = random.sample(com, iternum)
	selectk=[]
	for i in range(iternum):
		selectk.append(random.sample(rangek, comnum))
	
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
			for j in range(len(train_data[i])):
				temptraindata.append(train_data[i][j])
				temptraintarget.append(trainindex)
			trainindex=trainindex+1
		trainindex=trainindex-1

		temptestdata=[]
		temptesttarget=[]
		testindex=1
		for i in range(len(test_data)):
			for j in range(len(test_data[i])):
				temptestdata.append(test_data[i][j])
				temptesttarget.append(testindex)
			testindex=testindex+1		
		testindex=testindex-1
		
		print("训练集项目数：" ,trainindex)
		print("测试集项目数：",testindex)
		train_data=temptraindata
		train_target=temptraintarget
		test_data=temptestdata
		test_target=temptesttarget
		#将所需的不同手势类别划分为训练集和测试集
		train_data=np.array(train_data)
		test_data=np.array(test_data)
		print("train_data.shape:",train_data.shape)
		print("test_data.shape:",test_data.shape)

		#选择的锚数
		anchornum=3

		#将n*300变为300*n
		# temptraindata=[]
		# for i in range(len(train_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(train_data[0])):
		# 			temp[j].append(train_data[i][k][j])
		# 	temptraindata.append(temp)
		# train_data=temptraindata
		# temptestdata=[]
		# for i in range(len(test_data)):
		# 	temp=[]
		# 	for j in range(300):
		# 		temp.append([])
		# 		for k in range(len(test_data[0])):
		# 			temp[j].append(test_data[i][k][j])
		# 	temptestdata.append(temp)
		# test_data=temptestdata


		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)
		print("train_data.shape:",train_data.shape)
		print("test_data.shape:",test_data.shape)

		score,label= siamese_data(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
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




def siamese_mul_data_divide_class(feature,target,targetnum):

	#类别-样本-多个特征
	# 只有一个手势的情况
	maxusernum=targetnum
	print("数据集中的用户数：",maxusernum)
	tempfeature.append([] for i in range(maxusernum))
	for i in range(len(target)):
		tempfeature[int((target[i]-1))].append(feature[i])

	meanacc=[]
	meanfar=[]
	meanfrr=[]

	#循环次数
	iternum=30
	#组合内序号个数
	comnum=4
	#训练集个数
	# traincomnum=32

	rangek=list(range(0,maxusernum))
	#得出用户数在2之间的组合
	com=list(combinations(rangek,comnum))
	# #在组合间，随机选其中的iternum个
	selectk = random.sample(com, iternum)
	
	#若样本数多，则彻底随机化
	# selectk=[]
	# for i in range(iternum):
	# 	selectk.append(random.sample(rangek, comnum))

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
			for j in range(len(train_data[i])):
				temptraindata.append(train_data[i][j])
				temptraintarget.append(trainindex)
			trainindex=trainindex+1
		trainindex=trainindex-1

		temptestdata=[]
		temptesttarget=[]
		testindex=1
		for i in range(len(test_data)):
			for j in range(len(test_data[i])):
				temptestdata.append(test_data[i][j])
				temptesttarget.append(testindex)
			testindex=testindex+1		
		testindex=testindex-1
		
		print("训练集项目数：" ,trainindex)
		print("测试集项目数：",testindex)
		train_data=temptraindata
		train_target=temptraintarget
		test_data=temptestdata
		test_target=temptesttarget
		#将所需的不同手势类别划分为训练集和测试集
		train_data=np.array(train_data)
		test_data=np.array(test_data)
		print("train_data.shape:",train_data.shape)
		print("test_data.shape:",test_data.shape)

		featurenum=30
		anchornum=3

		#将n*300变为300*n
		temptraindata=[]
		for i in range(len(train_data)):
			temp=[]
			for j in range(300):
				temp.append([])
				for k in range(len(train_data[0])):
					temp[j].append(train_data[i][k][j])
			temptraindata.append(temp)
		train_data=temptraindata
		temptestdata=[]
		for i in range(len(test_data)):
			temp=[]
			for j in range(300):
				temp.append([])
				for k in range(len(test_data[0])):
					temp[j].append(test_data[i][k][j])
			temptestdata.append(temp)
		test_data=temptestdata


		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)
		print("train_data.shape:",train_data.shape)
		print("test_data.shape:",test_data.shape)

		score,label= siamese_mul_data(train_data,test_data, train_target, test_target,trainindex,testindex,featurenum,anchornum)
		score=[i[0] for i in score]
		label=[i for i in label]
		print('原结果：',label)
		print('预测分数：',score)
		accuracy,far,frr=cal_siamese_eer(label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfrr),")",)
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])





# def siamese_cwt_classifier(dataset,target,targetnum):
# 	meanacc=[]
# 	meanfar=[]
# 	meanfrr=[]	
# 	for t in range(0,1):
# 		train_data,test_data, train_target, test_target = train_test_split(dataset,target,test_size = 0.2,random_state = t*30,stratify=target)

# 		print("进入cwt处理阶段")	
# 		temptraindata=[]
# 		temptestdata=[]
# 		#提取1-25频段的时频图
# 		for i in range(len(train_data)):
# 			temp=[]
# 			for j in range(len(train_data[0])):
# 				coef, freqs=cwt(train_data[i][j],25,'mexh')
# 				temp.append(coef)
# 			temptraindata.append(temp)

# 		for i in range(len(test_data)):
# 			temp=[]
# 			for j in range(len(test_data[0])):
# 				coef, freqs=cwt(test_data[i][j],25,'mexh')
# 				temp.append(coef)
# 			temptestdata.append(temp)
# 		print("cwt处理完成")	
# 		train_data=temptraindata
# 		test_data=temptestdata


# 		# print("数据重排序(方案1)")	2*24*300到24*300*2
# 		temptraindata=[]
# 		temptestdata=[]
# 		for i in range(len(train_data)):
# 			temp=[]
# 			for j in range(24):
# 				temp.append([])
# 				for k in range(300):
# 					temp[j].append([])
# 					for d in range(len(train_data[0])):
# 						temp[j][k].append(train_data[i][d][j][k])
# 			temptraindata.append(temp)
# 		for i in range(len(test_data)):
# 			temp=[]
# 			for j in range(24):
# 				temp.append([])
# 				for k in range(300):
# 					temp[j].append([])
# 					for d in range(len(test_data[0])):
# 						temp[j][k].append(test_data[i][d][j][k])
# 			temptestdata.append(temp)
# 		train_data=temptraindata
# 		test_data=temptestdata

# 		# print("数据重排序(方案2)")	2*24*300到300*24*2

# 		# temptraindata=[]
# 		# temptestdata=[]
# 		# for i in range(len(train_data)):
# 		# 	temp=[]
# 		# 	for d in range(len(train_data[0])):
# 		# 		temp.append([])
# 		# 		for j in range(300):
# 		# 			temp[d].append([])
# 		# 			for k in range(24):
# 		# 				temp[d][j].append(train_data[i][d][k][j])
# 		# 	temptraindata.append(temp)
# 		# for i in range(len(test_data)):
# 		# 	temp=[]
# 		# 	for d in range(len(test_data[0])):
# 		# 		temp.append([])
# 		# 		for j in range(300):
# 		# 			temp[d].append([])
# 		# 			for k in range(24):
# 		# 				temp[d][j].append(test_data[i][d][k][j])
# 		# 	temptestdata.append(temp)
# 		# train_data=temptraindata
# 		# test_data=temptestdata


# 		train_data=np.array(train_data)
# 		train_target=np.array(train_target)
# 		test_data=np.array(test_data)
# 		test_target=np.array(test_target)

# 		# train_data=train_data.astype('float32')
# 		# test_data=test_data.astype('float32')

# 		#自己的方案
# 		score,test_label= siamese_cwt(train_data,test_data, train_target, test_target,targetnum)
# 		#cwt_emg方案
# 		# score,test_label,threshold= siamese_cwt_emg(train_data,test_data, train_target, test_target,targetnum)
# 		print(score)
# 		score=[i[0] for i in score]
# 		print('原结果：',test_label)
# 		print('预测分数：',score)
# 		i=0.001
# 		while i<5:
# 			tp,tn,fp,fn=siamese_accuracy_score(test_label,score,i)
# 			accuracy=(tp+tn)/(tp+tn+fp+fn)
# 			far=(fp)/(fp+tn)
# 			frr=(fn)/(fn+tp)
# 			# print(tp,tn,fp,fn)
# 			# print("i=",i)
# 			# print("accuracy:",accuracy,"far:",far,"frr:",frr)
# 			if frr<far:
# 				break
# 			i=i+0.005
# 		print("i=",i)
# 		print("accuracy:",accuracy,"far:",far,"frr:",frr)
# 		meanacc.append(accuracy)
# 		meanfar.append(far)
# 		meanfrr.append(frr)
# 	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))

# def siamese_ori_final_class(dataset,target,targetnum):
# 	dataset=np.array(dataset)
# 	target=np.array(target)
# 	score,label= siamese_ori_final(dataset,target,targetnum)
# 	print(score)
	
# 	score=[i[0] for i in score]
# 	label=[i for i in label]
# 	print('原结果：',label)
# 	print('预测分数：',score)
# 	i=0.1
# 	while i<3:
# 		tp,tn,fp,fn=siamese_accuracy_score(label,score,i)
# 		accuracy=(tp+tn)/(tp+tn+fp+fn)
# 		far=(fp)/(fp+tn)
# 		frr=(fn)/(fn+tp)
# 		if frr<far:
# 			break
# 		i=i+0.001
# 	print("i=",i)
# 	print("accuracy:",accuracy,"far:",far,"frr:",frr)