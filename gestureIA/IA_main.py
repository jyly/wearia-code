# -*- coding=utf-8 -*-
import sys
import wear_data_preprocess 
import phone_data_preprocess 
import other_phone_data_preprocess 
import filecontrol 
import sklearn_classifier
import siamese_classifier
import tripletloss_classifier
from itertools import combinations
import random
import numpy as np

if __name__ == "__main__":
	print(sys.version)
	#源数据路径
	wear_datadir='./oridata/'
	phone_datadir='./payIA_data/'

	#数据预处理
	# wear_data_preprocess.all_data(wear_datadir)#提取手势的具体数据段
	# wear_data_preprocess.all_feature(wear_datadir)#提取手势具体数据段的特征


	# 将预处理后的数据从文件读到内存
	# dataset,target,targetnum=filecontrol.dataread()
	feature,target,targetnum=filecontrol.featureread()

	print("总样本数：",len(target))
	print("总类别数：",targetnum)	
	print("总特征数：",len(feature[0]))


	# sklrean分类
	# sklearn_classifier.sklearn_mulclass(feature,target,targetnum)
	# sklearn_classifier.sklearn_finalmulclass(feature,target,targetnum)
	# sklearn_classifier.sklearn_oneclass(feature,target,1)
	# sklearn_classifier.sklearn_finaloneclass(feature,target,1)

	# siamese分类（基于原数据）
	# siamese_classifier.siamese_oridata_classifier(dataset[:,0:2],target,targetnum)
	# siamese_classifier.siamese_ori_final_class(dataset[:,0:2],target,targetnum)
	
	# siamese分类（基于连续小波）
	# siamese_classifier.siamese_cwt_classifier(dataset[:,0:2],target,targetnum)
	

	# siamese分类（基于特征）
	# siamese_classifier.siamese_feature_classifier(feature,target,targetnum)
	# siamese_classifier.siamese_feature_final_class(feature,target,targetnum)
	# siamese_classifier.siamese_feature_divide_class(feature,target,targetnum)
	# siamese_classifier.siamese_feature_divide_class(feature[:,0:88],target,targetnum)
	# siamese_classifier.siamese_feature_divide_class(feature[:,45:202],target,targetnum)
	# siamese_classifier.siamese_feature_divide_class(feature[:,88:],target,targetnum)
	# siamese_classifier.siamese_femotion_test_predature_inidivide_class(feature,target,targetnum)
	# siamese_classifier.siamese_mul_feature_divide_class(feature,target,targetnum)

	
	# siamese_classifier.siamese_feature_mul_build_class(feature,target,40,30)

	# siamese_classifier.siamese_feature_mul_final_class(feature,target,25,30,5)



	meanacc=[]
	meanfar=[]
	meanfrr=[]
	rangek=list(range(0,25))
	com=list(combinations(rangek,4))
	selectk = random.sample(com, 20)
	for t in range(20):

		print("被选择的测试集序号：",selectk[t])
		tempfeature=[]
		temptarget=[]
		for i in range(4):
			for j in range(len(target)):
				if target[j]==(selectk[t][i]+1):
					tempfeature.append(feature[j])
					temptarget.append(int(i+1))
		accuracy,far,frr=siamese_classifier.siamese_feature_mul_final_class(tempfeature,temptarget,4,30,5)
		meanacc.append(accuracy)
		meanfar.append(far)
		meanfrr.append(frr)
	print("meanacc:",np.mean(meanacc),"(",np.std(meanacc),")","meanfar:",np.mean(meanfar),"(",np.std(meanfar),")","meanfrr:",np.mean(meanfrr),"(",np.std(meanfrr),")",)
	# print("stdacc:",np.std(meanacc),"stdfar:",np.std(meanfar),"stdfrr:",np.mean(meanfrr))
	for i in range(len(meanacc)):
		print("被选择的测试集序号：",selectk[i])
		print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])

	# tripletloss
	# tripletloss_classifier.tripletloss_oridata_classifier(dataset[:,0:2],target,targetnum)
	# tripletloss_classifier.tripletloss_feature_classifier(feature,target,targetnum)
	# tripletloss_classifier.tripletloss_feature_divide_classifier(feature,target,targetnum)






	# 历史代码
	# other_phone_datadir='./other_phone_data/'
	# phone_data_preprocess.all_data(phone_datadir)
	# other_phone_data_preprocess.all_data(other_phone_datadir)
	# dataset,target,targetnum=filecontrol.phone_dataread()
