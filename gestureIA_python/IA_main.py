# -*- coding=utf-8 -*-
import filecontrol
import wear_data_preprocess
import gesture_detect_stistical

import siamese_data_classifier
import siamese_feature_classifier
import siamese_combine_classifier
import numpy as np
import tensorflow as tf
import keras_mulclassifier

if __name__ == "__main__":

	# 实验室服务器GPU使用设限
	# config = tf.compat.v1.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.8
	# session = tf.compat.v1.Session(config=config)

	#粗过滤手势检测率计算
	# onlygesture='./gesture_detect/onlygesture/'#存在手势的数据
	# nogesturedir='./gesture_detect/nogesture/'#前500个人中过滤出来的无手势数据
	# gesture_detect_stistical.detect_cal_true(onlygesture)
	# gesture_detect_stistical.detect_cal_false(nogesturedir)


	#python版数据预处理，最终提取的结果与java版存在差异
	# wear_datadir='./oridata/'
	# wear_data_preprocess.all_data(wear_datadir)#提取手势的具体数据段
	# wear_data_preprocess.all_feature(wear_datadir)#提取手势具体数据段的特征
	# wear_data_preprocess.renew_feature()#提取手势具体数据段的特征
	# print("总样本数：",len(target))
	# print("总类别数：",targetnum)
	# print("总特征数：",len(feature[0]))




	# siamese分类（基于原数据）
	sequence,target,targetnum=filecontrol.dataread()#将手势段直接作为特征输入
	# sequence,target,targetnum=filecontrol.same_gesture_selected(sequence,target,targetnum,3)#选择同一手势的数据

	#40选8进行标准认证
	# siamese_data_classifier.siamese_data_authentication(sequence[:,:2],target,targetnum)
	#多任务模型的识别和认证
	# siamese_data_classifier.siamese_data_multask(sequence[:,:2],target,targetnum)

	# siamese_data_classifier.siamese_data_build_class(sequence[:,:2],target,targetnum)
	siamese_data_classifier.siamese_data_test_class(sequence[:,:2],target,targetnum,5)
	# keras_mulclassifier.keras_mulclass(sequence[:,:2],target,targetnum)





	# 将提取的特征从文件读到内存
	# siamese分类（基于特征)
	# feature,target,targetnum=filecontrol.featureread()#人工利用统计量提取特征
	# feature,target,targetnum=filecontrol.same_gesture_selected(feature,target,targetnum,3)

	# tempfeature=[]
	# for i in range(len(feature)):
	# 	temp=[]
	# 	for j in range(30):
	# 		temp.append(feature[i][j])
	# 	for j in range(30):
	# 		temp.append(feature[i][j+101])
	# 	tempfeature.append(temp)
	# tempfeature=np.array(tempfeature)
	# print(tempfeature.shape)

	# tempfeature=[]
	# for i in range(len(feature)):
	# 	temp=[]
	# 	for j in range(70):
	# 		temp.append(feature[i][j+31])
	# 	for j in range(70):
	# 		temp.append(feature[i][j+131])
	# 	tempfeature.append(temp)
	# tempfeature=np.array(tempfeature)
	# print(tempfeature.shape)

	# siamese_feature_classifier.siamese_feature_mul_class(feature,target,targetnum)
	# siamese_feature_classifier.siamese_feature_authentication(feature[:,:202],target,targetnum)
	# siamese_feature_classifier.siamese_feature_authentication(tempfeature,target,targetnum)




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
