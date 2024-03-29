# -*- coding=utf-8 -*-
import numpy as np
from siamese.siamese_data_model import *
from classifier_tool import *
from normal_tool import *
import IAtool


def siamese_data_authentication(sequence,target,targetnum):

	meanacc=[]
	meanfar=[]
	meanfrr=[]
	#选择的锚数
	anchornum=3
	#循环次数
	iternum=10
	#组合内序号个数
	testsetnumber=8
	#训练集个数
	# traincomnum=28
	selectk=IAtool.create_rank_testnum(targetnum,iternum,testsetnumber)

	# 写死选择的数据集，方便比较
	# selectk=[]
	# for i in range(iternum):
	# 	temp=[]
	# 	startpoint=(i*7)%40
	# 	for j in range(testsetnumber):
	# 		temp.append((startpoint+j*2)%40)
	# 	selectk.append(temp)	

	tempfeature=IAtool.listtodic(sequence,target)

	for t in range(iternum):
		print("周期：",t)
		train_data,test_data,train_target,test_target,trainindex,testindex=IAtool.allot_data(selectk[t],targetnum,tempfeature)

		score,label= siamese_authentication(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
		# score,label= siamese_mul_model_data(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
		score=[i[0] for i in score]
		label=[i for i in label]
		print('原结果：',label[:50])
		print('预测分数：',score[:50])
		accuracy,far,frr=cal_siamese_eer(label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfar),")",)
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i]) 
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])

def siamese_data_build_class(train_data,train_target,trainindex):
	# train_data=IAtool.datatranspose(train_data)
	siamese_data_build(train_data,train_target,trainindex)

def siamese_data_test_class(test_data,test_target,targetnum,anchornum):
	# test_data=IAtool.datatranspose(test_data)
	score,label= siamese_data_test(test_data,test_target,targetnum,anchornum)
	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label[:50])
	print('预测分数：',score[:50])
	accuracy,far,frr=cal_siamese_eer(label,score)
	print("acc:",accuracy,"far:",far,"frr:",frr)
	return accuracy,far,frr




#多任务孪生网络，同时识别手势和用户
# def siamese_data_multask(feature,target,targetnum):
# 	tempfeature=IAtool.listtodic(feature,target)
# 	gesturemeanacc=[]
# 	gesturemeanfar=[]
# 	gesturemeanfrr=[]
# 	usermeanacc=[]
# 	usermeanfar=[]
# 	usermeanfrr=[]
# 	#选择的锚数
# 	anchornum=3
# 	#循环次数
# 	iternum=10
# 	#组合内序号个数
# 	testsetnumber=8
# 	rangek=list(range(0,targetnum-1))
# 	selectk=[]
# 	for i in range(iternum):
# 		temp=[]
# 		startpoint=i*9
# 		for j in range(18):
# 			temp.append((startpoint+j)%40)
# 		selectk.append(temp)	
# 	for t in range(iternum):
# 		print("周期：",t)
# 		train_data=[]
# 		test_data=[]

# 		print("被选择的测试集序号：",selectk[t])
# 		for i in range(targetnum):
# 			if i in selectk[t]:
# 				test_data.append(tempfeature[i])
# 			else:
# 				train_data.append(tempfeature[i])	
# 		train_data,train_target,trainindex=IAtool.dictolist(train_data)
# 		test_data,test_target,testindex=IAtool.dictolist(test_data)
# 		print("训练集项目数：" ,trainindex)
# 		print("测试集项目数：",testindex)
# 		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)
# 		score,label= siamese_mul_task(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
# 		score=[[i[0] for i in score[0]],[i[0] for i in score[1]]]
# 		print('预测分数：',score[0][:90])
# 		useraccuracy,userfar,userfrr=cal_siamese_eer(label[0],score[0])
# 		usermeanacc.append(useraccuracy)
# 		usermeanfar.append(userfar)
# 		usermeanfrr.append(userfrr)
# 		gestureaccuracy,gesturefar,gesturefrr=cal_siamese_eer(label[1],score[1])
# 		gesturemeanacc.append(gestureaccuracy)
# 		gesturemeanfar.append(gesturefar)
# 		gesturemeanfrr.append(gesturefrr)
# 	print("usermeanacc:",np.mean(usermeanacc),"(",np.std(usermeanacc),")","usermeanfar:",np.mean(usermeanfar),"(",np.std(usermeanfar),")","userfrr:",np.mean(userfrr),"(",np.std(userfrr),")",)
# 	print("gesturemeanacc:",np.mean(gesturemeanacc),"(",np.std(gesturemeanacc),")","gesturemeanfar:",np.mean(gesturemeanfar),"(",np.std(gesturemeanfar),")","gesturemeanfrr:",np.mean(gesturemeanfrr),"(",np.std(gesturemeanfrr),")",)
# 	for i in range(len(usermeanacc)):
# 		print("被选择的测试集序号：",selectk[i])
# 		print("usermeanacc:",usermeanacc[i],"usermeanfar:",usermeanfar[i],"usermeanfrr:",usermeanfrr[i])
# 		print("gesturemeanacc:",gesturemeanacc[i],"far:",gesturemeanfar[i],"gesturemeanfrr:",gesturemeanfrr[i])









#多传感器输入的孪生网络分类
# def siamese_mul_data_divide_class(feature,target,targetnum):
# 	tempfeature=IAtool.listtodic(feature,target)
# 	meanacc=[]
# 	meanfar=[]
# 	meanfrr=[]
# 	#循环次数
# 	iternum=10
# 	#组合内序号个数
# 	testsetnumber=8
# 	rangek=list(range(0,targetnum-1))
# 	selectk=[]
# 	for t in range(iternum):
# 		selectk.append(random.sample(rangek, testsetnumber))
# 	for t in range(iternum):
# 		print("周期：",t)
# 		train_data=[]
# 		test_data=[]
# 		print("被选择的测试集序号：",selectk[t])
# 		for i in range(targetnum):
# 			if i in selectk[t]:
# 				test_data.append(tempfeature[i])
# 			# if i in selectks:
# 			else:
# 				train_data.append(tempfeature[i])	
# 		train_data,train_target,trainindex=IAtool.dictolist(train_data)
# 		test_data,test_target,testindex=IAtool.dictolist(test_data)
# 		print("训练集项目数：" ,trainindex)
# 		print("测试集项目数：",testindex)
# 		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)
# 		anchornum=3

# 		#将n*300变为300*n
# 		temptraindata=[]
# 		for i in range(len(train_data)):
# 			temp=[]
# 			for j in range(200):
# 				temp.append([])
# 				for k in range(len(train_data[0])):
# 					temp[j].append(train_data[i][k][j])
# 			temptraindata.append(temp)
# 		train_data=temptraindata
# 		temptestdata=[]
# 		for i in range(len(test_data)):
# 			temp=[]
# 			for j in range(200):
# 				temp.append([])
# 				for k in range(len(test_data[0])):
# 					temp[j].append(test_data[i][k][j])
# 			temptestdata.append(temp)
# 		test_data=temptestdata

# 		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)

# 		score,label= siamese_mul_data(train_data,test_data, train_target, test_target,trainindex,testindex,featurenum,anchornum)
# 		score=[i[0] for i in score]
# 		label=[i for i in label]
# 		print('原结果：',label)
# 		print('预测分数：',score)
# 		accuracy,far,frr=cal_siamese_eer(label,score)
# 		meanacc.append(accuracy)
# 		meanfar.append(far)
# 		meanfrr.append(frr)
# 	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfrr),")",)
# 	for i in range(len(meanacc)):
# 		print("被选择的测试集序号：",selectk[i])
# 		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])




#离散小波处理的孪生网络分类
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
