# -*- coding=utf-8 -*-
import filecontrol 
import wear_data_preprocess 
import gesture_detect_stistical

import siamese_data_classifier
import siamese_feature_classifier
import siamese_combine_classifier

import tensorflow as tf

if __name__ == "__main__":

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	session = tf.compat.v1.Session(config=config)
	#粗过滤手势检测率计算
	# onlygesture='./gesture_detect/onlygesture/'#存在手势的数据
	# nogesturedir='./gesture_detect/nogesture/'#前500个人眼过滤过的无手势数据
	# gesture_detect_stistical.detect_cal_true(onlygesture)
	# gesture_detect_stistical.detect_cal_false(nogesturedir)


	#python版数据预处理，java提取特征更快
	# wear_datadir='./oridata/'
	# wear_data_preprocess.all_data(wear_datadir)#提取手势的具体数据段
	# wear_data_preprocess.all_feature(wear_datadir)#提取手势具体数据段的特征
	# wear_data_preprocess.renew_feature()#提取手势具体数据段的特征


	# print("总样本数：",len(target))
	# print("总类别数：",targetnum)	
	# print("总特征数：",len(feature[0]))

	# 将预处理后的数据从文件读到内存
	# siamese分类（基于原数据）
	feature,target,targetnum=filecontrol.dataread()#将手势段直接作为特征输入
	# feature,target,targetnum=filecontrol.same_gesture_selected(feature,target,targetnum,3)

	# siamese_data_classifier.siamese_data_class(feature,target,targetnum)
	siamese_data_classifier.siamese_data_class(feature[:,:2],target,targetnum)






	# 将提取的特征从文件读到内存
	# siamese分类（基于特征)
	# feature,target,targetnum=filecontrol.featureread()#人工利用统计量提取特征
	# feature,target,targetnum=filecontrol.same_gesture_selected(feature,target,targetnum,3)

	# feature=np.array(feature)
	# tempfeature3=[]
	# for i in range(len(feature)):
	# 	temp=[]
	# 	for j in range(30):
	# 		temp.append(feature[i][j])
	# 	for j in range(30):
	# 		temp.append(feature[i][j+101])
	# 	tempfeature3.append(temp)
	# tempfeature3=np.array(tempfeature3)
	# print(tempfeature3.shape)

	# tempfeature3=[]
	# for i in range(len(feature)):
	# 	temp=[]
	# 	for j in range(70):
	# 		temp.append(feature[i][j+31])
	# 	for j in range(70):
	# 		temp.append(feature[i][j+131])
	# 	tempfeature3.append(temp)
	# tempfeature3=np.array(tempfeature3)
	# print(tempfeature3.shape)

	# siamese_feature_classifier.siamese_feature_mul_class(feature,target,targetnum)
	# siamese_feature_classifier.siamese_feature_class(feature[:,:202],target,targetnum)
	# siamese_feature_classifier.siamese_feature_class(tempfeature3,target,targetnum)

	





	


	#siamese分类（特征和原数据的组合）
	# data,target,targetnum=filecontrol.dataread()#将手势段直接作为特征输入
	# print(len(target))
	# print(targetnum)
	# feature,target,targetnum=filecontrol.featureread()#人工利用统计量提取特征
	# print(len(target))
	# print(targetnum)
	# data=np.array(data)
	# feature=np.array(feature)
	# print(data.shape,feature.shape)
	# data,datatarget,datatargetnum=filecontrol.same_gesture_selected(data,target,targetnum,3)
	# feature,target,targetnum=filecontrol.same_gesture_selected(feature,target,targetnum,3)
	# print(data.shape,feature.shape)

	# print("总样本数：",len(target))
	# print("总类别数：",targetnum)	
	# print("总特征数：",len(feature[0]))
	# print("总样本长数：",len(data[0][0]))

	# siamese_combine_classifier.siamese_combine_class(data,feature,target,targetnum)


# import tripletloss_classifier

	# tripletloss
	# tripletloss_classifier.tripletloss_oridata_classifier(dataset[:,0:2],target,targetnum)
	# tripletloss_classifier.tripletloss_feature_classifier(feature,target,targetnum)
	# tripletloss_classifier.tripletloss_feature_divide_classifier(feature,target,targetnum)








# import sklearn_classifier



	# sklrean分类
	# sklearn_classifier.sklearn_mulclass(feature[:,:76],target,targetnum)
	# sklearn_classifier.sklearn_build_class(feature[:,:76],target,targetnum)
	# sklearn_classifier.sklearn_final_class(feature[:,:76],target,targetnum)
	# sklearn_classifier.sklearn_TMC_mulclass(feature[:,:2],target,targetnum)



	# sklearn_classifier.sklearn_mulclass(feature[:,76:],target,targetnum)
	# sklearn_classifier.sklearn_build_class(feature[:,76:],target,targetnum)
	# sklearn_classifier.sklearn_final_class(feature[:,76:],target,targetnum)


	


	# 历史代码 import phone_data_preprocess  import other_phone_data_preprocess
	# other_phone_datadir='./other_phone_data/'
	# phone_data_preprocess.all_data(phone_datadir)
	# other_phone_data_preprocess.all_data(other_phone_datadir)
	# dataset,target,targetnum=filecontrol.phone_dataread()
