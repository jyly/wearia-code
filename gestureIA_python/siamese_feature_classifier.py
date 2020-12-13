# -*- coding=utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from siamese.siamese_feature_model import *
from classifier_tool import *
from normal_tool import *
import IAtool
import random
from itertools import combinations


#对自己提取的特征分类
def siamese_feature_based_classifier(dataset,target,targetnum):
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
		# IAtool.filterparameterwrite(sort,lda_bar,lda_scaling,'./parameter/ldapropara.txt')
		# train_data,test_data,pca_mean,pca_components=IAtool.pcapro(train_data,test_data)
		# IAtool.filterparameterwrite(sort,pca_mean,pca_components,'./parameter/pcapropara.txt')
		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)
		# IAtool.filterparameterwrite(sort,scale_mean,scale_scale,'./parameter/stdpropara.txt')

		train_data=np.array(train_data)
		train_target=np.array(train_target)
		test_data=np.array(test_data)
		test_target=np.array(test_target)
		print("训练集样本数：",train_data.shape)
		print("测试集样本数：",test_data.shape)

		score,test_label= siamese_feature(train_data,test_data, train_target, test_target,targetnum,targetnum,3)
		print('原结果：',test_label)
		score=[i[0] for i in score]
		print('预测分数：',score)
		accuracy,far,frr=cal_siamese_eer(test_label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfar),")",)
	for i in range(len(meanacc)):
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])





def siamese_feature_divide_class(feature,target,targetnum):

	tempfeature=IAtool.listtodic(feature,target)

	meanacc=[]
	meanfar=[]
	meanfrr=[]

	#循环次数
	iternum=30
	#组合内序号个数
	testsetnumber=8
	#训练集个数
	# trainsetnumber=32

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

		# 用于限制训练集数量
		selectks=[]
		for i in rangek:
			if i not in selectk[t]:
				selectks.append(i)
		selectks = random.sample(selectks, trainsetnumber)
		print("被选择的训练集序号：",selectks)

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
		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)


		#特征数，选择的锚数
		featurenum=10
		anchornum=5

		# sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
		# train_data=IAtool.scoreselect(train_data,sort,featurenum)
		# test_data=IAtool.scoreselect(test_data,sort,featurenum)
		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,featurenum)
		
		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)

		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)

		score,label= siamese_feature(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
		score=[i[0] for i in score]
		label=[i for i in label]
		print('原结果：',label)
		print('预测分数：',score)
		siamese_far_frr(label,score)
		accuracy,far,frr=cal_siamese_eer(label,score)
		# siamese_feature_build_class(train_data,train_target,trainindex,featurenum)
		# accuracy,far,frr=siamese_feature_final_class(test_data,test_target,testindex,featurenum,anchornum)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfar),")",)
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])


def siamese_feature_mul_divide_class(feature,target,targetnum):

	tempfeature=IAtool.listtodic(feature,target)

	meanacc=[]
	meanfar=[]
	meanfrr=[]

	#循环次数
	iternum=10
	#组合内序号个数
	testsetnumber=8
	#训练集个数
	# trainsetnumber

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
		# selectks = random.sample(selectks, trainsetnumber)
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

		
		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)

		featurenum=30
		anchornum=3

		ppg_train_data,ppg_test_data,ppg_sort=IAtool.minepro(train_data[:,0:76],test_data[:,0:76],train_target,featurenum)
		motion_train_data,motion_test_data,motion_sort=IAtool.minepro(train_data[:,76:],test_data[:,76:],train_target,featurenum)
		
		# ppg_sort,motion_sort,scale_mean,scale_scale=IAtool.mulfilterparameterread('./parameter/stdpropara.txt')
		# scale_mean=np.array(scale_mean)
		# scale_scale=np.array(scale_scale)
		# ppg_train_data=IAtool.scoreselect(train_data[:,0:76],ppg_sort,featurenum)
		# motion_train_data=IAtool.scoreselect(train_data[:,76:],motion_sort,featurenum)
		# ppg_test_data=IAtool.scoreselect(test_data[:,0:76],ppg_sort,featurenum)
		# motion_test_data=IAtool.scoreselect(test_data[:,76:],motion_sort,featurenum)

		train_data=IAtool.datacombine(ppg_train_data,motion_train_data)
		test_data=IAtool.datacombine(ppg_test_data,motion_test_data)
	

		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)
		# IAtool.mulfilterparameterwrite(ppg_sort,motion_sort,scale_mean,scale_scale,'./stdpropara.txt')

		train_data,test_data,train_target,test_target=IAtool.datashape(train_data,test_data,train_target,test_target)

		# score,label= siamese_mul_feature(train_data,test_data, train_target, test_target,trainindex,testindex,featurenum,anchornum)
		score,label= siamese_mul_model_feature(train_data,test_data, train_target, test_target,trainindex,testindex,featurenum,anchornum)
		score=[i[0] for i in score]
		label=[i for i in label]
		print('原结果：',label)
		print('预测分数：',score)
		siamese_far_frr(label,score)
		accuracy,far,frr=cal_siamese_eer(label,score)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfrr),")",)
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])



#对未知数据进行目标分类模型的训练
def siamese_feature_build_class(train_data,train_target,trainindex,featurenum):
	##训练降维模型
	temp=[]
	# train_data,temp,sort=IAtool.minepro(train_data,temp,train_target,featurenum)
	# sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
	train_data,temp,scale_mean,scale_scale=IAtool.stdpro(train_data,temp)
	IAtool.filterparameterwrite(sort,scale_mean,scale_scale,'./parameter/stdpropara.txt')

	#读入文件加快特征降维速度
	# sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
	# scale_mean=np.array(scale_mean)
	# scale_scale=np.array(scale_scale)
	# train_data=IAtool.scoreselect(train_data,sort,featurenum)
	# train_data=(train_data-scale_mean)/scale_scale

	train_data=np.array(train_data)
	train_target=np.array(train_target)
	print("train_data.shape:",train_data.shape)
	siamese_feature_buildmodel(train_data,train_target,trainindex)

#利用训练好的模型进行训练
def siamese_feature_final_class(test_data,test_target,targetnum,featurenum,anchornum):

	sort,scale_mean,scale_scale=IAtool.filterparameterread('./parameter/stdpropara.txt')
	scale_mean=np.array(scale_mean)
	scale_scale=np.array(scale_scale)
	test_data=np.array(test_data)
	test_target=np.array(test_target)

	test_data=IAtool.scoreselect(test_data,sort,featurenum)
	test_data=(test_data-scale_mean)/scale_scale

	score,label= siamese_feature_final(test_data,test_target,targetnum,anchornum)

	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	accuracy,far,frr=cal_siamese_eer(label,score)
	return accuracy,far,frr





def siamese_feature_mul_build_class(train_data,train_target,trainindex,featurenum):
	#加载已有信息熵和标准差
	ppg_sort,motion_sort,scale_mean,scale_scale=IAtool.mulfilterparameterread('./parameter/stdpropara.txt')
	scale_mean=np.array(scale_mean)
	scale_scale=np.array(scale_scale)

	ppg_train_data=IAtool.scoreselect(train_data[:,0:76],ppg_sort,featurenum)
	motion_train_data=IAtool.scoreselect(train_data[:,76:],motion_sort,featurenum)

	train_data=[]
	for i in range(len(ppg_train_data)):
		temp=[]
		for j in ppg_train_data[i]:
			temp.append(j)
		for j in motion_train_data[i]:
			temp.append(j)
		train_data.append(temp)
	# train_data=(train_data-scale_mean)/scale_scale
	temps=[]
	train_data,temps,scale_mean,scale_scale=IAtool.stdpro(train_data,temps)
	IAtool.mulfilterparameterwrite(ppg_sort,motion_sort,scale_mean,scale_scale,'./parameter/stdpropara.txt')
	train_data=np.array(train_data)
	train_target=np.array(train_target)
	print("train_data.shape:",train_data.shape)
	print("train_target.shape:",train_target.shape)

	# 重新训练信息熵和标准差
	# temps=[]
	# ppg_train_data,temps,ppg_sort=IAtool.minepro(train_data[:,0:76],temps,train_target,featurenum)
	# motion_train_data,temps,motion_sort=IAtool.minepro(train_data[:,76:],temps,train_target,featurenum)
	# train_data=[]
	# for i in range(len(ppg_train_data)):
	# 	temp=[]
	# 	for j in ppg_train_data[i]:
	# 		temp.append(j)
	# 	for j in motion_train_data[i]:
	# 		temp.append(j)
	# 	train_data.append(temp)
	# train_data,temps,scale_mean,scale_scale=IAtool.stdpro(train_data,temps)
	# IAtool.mulfilterparameterwrite(ppg_sort,motion_sort,scale_mean,scale_scale,'./parameter/stdpropara.txt')
	# train_data=np.array(train_data)
	# train_target=np.array(train_target)
	# print("train_data.shape:",train_data.shape)

	siamese_mul_feature_buildmodel(train_data,train_target,trainindex,featurenum)

def siamese_feature_mul_final_class(test_data,test_target,targetnum,featurenum,anchornum):
	ppg_sort,motion_sort,scale_mean,scale_scale=IAtool.mulfilterparameterread('./parameter/stdpropara.txt')
	scale_mean=np.array(scale_mean)
	scale_scale=np.array(scale_scale)
	test_data=np.array(test_data)
	test_target=np.array(test_target)

	ppg_test_data=IAtool.scoreselect(test_data[:,0:76],ppg_sort,featurenum)
	motion_test_data=IAtool.scoreselect(test_data[:,76:],motion_sort,featurenum)

	test_data=[]
	for i in range(len(ppg_test_data)):
		temp=[]
		for j in ppg_test_data[i]:
			temp.append(j)
		for j in motion_test_data[i]:
			temp.append(j)
		test_data.append(temp)
	
	# print(test_data[0])
	test_data=(test_data-scale_mean)/scale_scale
	# print(test_data[0])
	test_data=np.array(test_data)
	test_target=np.array(test_target)
	print("test_data.shape:",test_data.shape)
	print("test_target.shape:",test_target.shape)

	score,label= siamese_mul_feature_final(test_data,test_target,targetnum,featurenum,anchornum)
	score=[i[0] for i in score]
	label=[i for i in label]
	print('原结果：',label)
	print('预测分数：',score)
	accuracy,far,frr=cal_siamese_eer(label,score)
	return accuracy,far,frr



def siamese_feature_mul_single(train_data,test_data,featurenum):
	train_data=np.array(train_data)
	test_data=np.array(test_data)
	ppg_sort,motion_sort,scale_mean,scale_scale=IAtool.mulfilterparameterread('./parameter/stdpropara.txt')
	scale_mean=np.array(scale_mean)
	scale_scale=np.array(scale_scale)


	ppg_train_data=IAtool.scoreselect(train_data[:,0:252],ppg_sort,featurenum)
	motion_train_data=IAtool.scoreselect(train_data[:,252:],motion_sort,featurenum)
	train_data=[]
	for i in range(len(ppg_train_data)):
		temp=[]
		for j in ppg_train_data[i]:
			temp.append(j)
		for j in motion_train_data[i]:
			temp.append(j)
		train_data.append(temp)
	train_data=(train_data-scale_mean)/scale_scale

	ppg_test_data=IAtool.scoreselect(test_data[:,0:252],ppg_sort,featurenum)
	motion_test_data=IAtool.scoreselect(test_data[:,252:],motion_sort,featurenum)
	test_data=[]
	for i in range(len(ppg_test_data)):
		temp=[]
		for j in ppg_test_data[i]:
			temp.append(j)
		for j in motion_test_data[i]:
			temp.append(j)
		test_data.append(temp)
	test_data=(test_data-scale_mean)/scale_scale
	print("test_data.shape:",test_data.shape)

	score= siamese_mul_feature_single(train_data,test_data,featurenum)
	# score=[i[0] for i in score]
	print('预测分数：',score)
	# accuracy,far,frr=cal_siamese_eer(label,score)
	# return accuracy,far,frr




#多线程代码，后期莫名其妙跑不掉，不知那里有bug，待查
# def eercal(label_score,finalacc,iternum):
# 	print('计算EER进程: %s' % os.getpid())
# 	while finalacc.qsize()!=(iternum+1):

# 		# if finalacc.qsize()==iternum:
# 			# break
# 		if label_score.qsize()>0:
# 			score,label=label_score.get()
# 			score=[i[0] for i in score]
# 			label=[i for i in label]
# 			print('原结果：',label)
# 			print('预测分数：',score)
# 			i=0.01
# 			while i<3:
# 				tp,tn,fp,fn=siamese_accuracy_score(label,score,i)
# 				accuracy=(tp+tn)/(tp+tn+fp+fn)
# 				far=(fp)/(fp+tn)
# 				frr=(fn)/(fn+tp)
# 				# print("i=",i)
# 				# print("accuracy:",accuracy,"far:",far,"frr:",frr)
# 				if frr<far:
# 					break
# 				i=i+0.01
# 			print("i=",i)
# 			print("accuracy:",accuracy,"far:",far,"frr:",frr)
# 			finalacc.put([accuracy,far,frr])
# 			print("finalacc.qsize():",finalacc.qsize())
# 	print('eercal is over: %s' % os.getpid())


# def modeltrain(dataset,label_score,anchornum,finalacc,iternum):
# 	print('训练模型并得出测试集结果: %s' % os.getpid())
# 	while finalacc.qsize()!=iternum:
# 		# if finalacc.qsize()==iternum:
# 			# break
# 		if dataset.qsize()>0 and label_score.qsize()<1:
# 			train_data,test_data,train_target,test_target,trainindex,testindex=dataset.get()
# 			score,label= siamese_feature(train_data,test_data, train_target, test_target,trainindex,testindex,anchornum)
# 			label_score.put([score,label])
# 	print('modeltrain is over: %s' % os.getpid())



# def datapre(dataset,tempfeature,selectk):
# 	print('构建测试对和训练对: %s' % os.getpid())
# 	iternum=len(selectk)
# 	t=0
# 	while t!=iternum:
# 		# if t==iternum:
# 			# break
# 		# if dataset.qsize()<1:
# 		print("周期：",t)
# 		train_data=[]
# 		test_data=[]
# 		print("被选择的测试集序号：",selectk[t])
# 		for i in range(len(tempfeature)):
# 			if i in selectk[t]:
# 				test_data.append(tempfeature[i])
# 			# if i in selectks:
# 			else:
# 				train_data.append(tempfeature[i])	
# 		t=t+1
# 		print("t:",t)

# 		temptraindata=[]
# 		temptraintarget=[]
# 		trainindex=1
# 		for i in range(len(train_data)):
# 			for j in range(8,9):
# 				for k in range(len(train_data[i][j])):
# 					temptraindata.append(train_data[i][j][k])
# 					temptraintarget.append(trainindex)
# 				trainindex=trainindex+1
# 		trainindex=trainindex-1
# 		print("训练集项目数：" ,trainindex)

# 		temptestdata=[]
# 		temptesttarget=[]
# 		testindex=1
# 		for i in range(len(test_data)):
# 			for j in range(8,9):
# 				for k in range(len(test_data[i][j])):
# 					temptestdata.append(test_data[i][j][k])
# 					temptesttarget.append(testindex)
# 				testindex=testindex+1		
# 		testindex=testindex-1
# 		print("测试集项目数：",testindex)
# 		train_data=temptraindata
# 		train_target=temptraintarget
# 		test_data=temptestdata
# 		test_target=temptesttarget
		
# 		featurenum=30
		

# 		train_data,test_data,sort=IAtool.minepro(train_data,test_data,train_target,featurenum)
# 		# train_data,temp,lda_bar,lda_scaling=IAtool.ldapro(train_data,temp,train_target)
# 		# IAtool.filterparameterwrite(sort,lda_bar,lda_scaling,'./ldapropara.txt')
		
# 		# train_data,temp,pca_mean,pca_components=IAtool.pcapro(train_data,temp)
# 		# IAtool.filterparameterwrite(sort,pca_mean,pca_components,'./pcapropara.txt')
# 		train_data,test_data,scale_mean,scale_scale=IAtool.stdpro(train_data,test_data)
# 		train_data=np.array(train_data)
# 		train_target=np.array(train_target)
# 		test_data=np.array(test_data)
# 		test_target=np.array(test_target)

# 		print("train_data.shape:",train_data.shape)
# 		print("train_data.shape:",test_data.shape)


		
# 		dataset.put([train_data,test_data,train_target,test_target,trainindex,testindex])
# 	print('datapre is over: %s' % os.getpid())



# def siamese_feature_inidivide_class(feature,target,targetnum):
# 	#将特征分为不同的人和手势类
# 	tempfeature=[]
# 	maxusernum=int(targetnum/9)
# 	print("数据集中的用户数：",maxusernum)
# 	for i in range(maxusernum):
# 		tempfeature.append([])
# 		for j in range(9):
# 			tempfeature[i].append([])
# 	for i in range(len(target)):
# 		t1=int((target[i]-1)/9)
# 		t2=(target[i]-1)%9
# 		tempfeature[t1][t2].append(feature[i])
# 	meanacc=[]
# 	meanfar=[]
# 	meanfrr=[]

# 	#循环次数
# 	iternum=5
# 	#组合内序号个数
# 	comnum=8

# 	rangek=list(range(0,maxusernum))
# 	#得出用户数在2之间的组合
# 	com=list(combinations(rangek,comnum))
# 	#在组合间，随机选其中的iternum个
# 	selectk = random.sample(com, iternum)
# 	anchornum=5

# 	print("进入多进程模式")
# 	#测试对和训练对的序列
# 	dataset = Queue()
# 	#测试集的标签和分数
# 	label_score = Queue()
# 	#最终训练分数
# 	finalacc= Queue()
# 	#数据预处理进程
# 	dataprepro = Process(target=datapre, args=(dataset,tempfeature,selectk))
# 	#模型训练及得出测试集结果进程
# 	modeltrainpro = Process(target=modeltrain, args=(dataset,label_score,anchornum,finalacc,iternum))
# 	#计算EER进程
# 	eercalpro1 = Process(target=eercal, args=(label_score,finalacc,iternum))

# 	dataprepro.start()
# 	modeltrainpro.start()
# 	eercalpro1.start()
# 	# eercalpro2.start()

# 	dataprepro.join()
# 	modeltrainpro.join()
# 	eercalpro1.join()
# 	# eercalpro2.join()

# 	for i in range(finalacc.qsize()):
# 		accuracy,far,frr=finalacc.get()
# 		meanacc.append(accuracy)
# 		meanfar.append(far)
# 		meanfrr.append(frr)
# 		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])
# 	print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))

