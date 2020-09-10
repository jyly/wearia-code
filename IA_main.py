# -*- coding=utf-8 -*-
import sys
import wear_data_preprocess 
import phone_data_preprocess 
import other_phone_data_preprocess 
import filecontrol 
import sklearn_classifier
import siamese_classifier
import tripletloss_classifier

if __name__ == "__main__":
	print(sys.version)
	#源数据路径
	wear_datadir='./oridata/'
	phone_datadir='./payIA_data/'
	other_phone_datadir='./other_phone_data/'

	#数据预处理
	# wear_data_preprocess.all_data(wear_datadir)
	wear_data_preprocess.all_feature(wear_datadir)
	# # phone_data_preprocess.all_data(phone_datadir)
	# other_phone_data_preprocess.all_data(other_phone_datadir)


	# 将预处理后的数据从文件读到内存==
	# dataset,target,targetnum=filecontrol.dataread()
	feature,target,targetnum=filecontrol.featureread()
	# dataset,target,targetnum=filecontrol.phone_dataread()

	# print("总样本数：",len(target))
	# print("总类别数：",targetnum)


	# sklrean分类
	# sklearn_classifier.sklearn_mulclass(feature,target,targetnum)
	# sklearn_classifier.sklearn_finalmulclass(feature,target,targetnum)
	# sklearn_classifier.sklearn_oneclass(feature,target,1)
	# sklearn_classifier.sklearn_finaloneclass(feature,target,1)

	# siamese分类（基于原数据）
	# siamese_classifier.siamese_oridata_classifier(dataset[:,0:2],target,targetnum)
	# siamese_classifier.siamese_oridata_classifier(dataset[:,0:8],target,targetnum)
	# siamese_classifier.siamese_ori_final_class(dataset[:,0:2],target,targetnum)
	
	# siamese分类（基于连续小波）
	# siamese_classifier.siamese_cwt_classifier(dataset[:,0:2],target,targetnum)
	

	# siamese分类（基于特征）
	# siamese_classifier.siamese_feature_classifier(feature,target,targetnum)
	# siamese_classifier.siamese_feature_final_class(feature,target,targetnum)
	siamese_classifier.siamese_feature_divide_class(feature,target,targetnum)

	# tripletloss
	# tripletloss_classifier.tripletloss_oridata_classifier(dataset[:,0:2],target,targetnum)
	# tripletloss_classifier.tripletloss_feature_classifier(feature,target,targetnum)
	# tripletloss_classifier.tripletloss_feature_divide_classifier(feature,target,targetnum)