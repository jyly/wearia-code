# -*- coding=utf-8 -*-

#计算多分类的准确度
def mul_accuracy_result(target,result,divnum):#目标，结果，目标对象的数量
	tp=0
	tn=0
	fp=0
	fn=0

	for i in range(1,divnum+1):
		for j in range(len(result)):
			if result[j]==i:
				if target[j]==i:
					tp=tp+1
				else:
					fp=fp+1
			else:
				if target[j]==i:
					fn=fn+1
				else:
					tn=tn+1
	return tp,tn,fp,fn

#计算多分类的准确度
def mul_accuracy_score(target,scores,threshold,divnum):#目标，结果，目标对象的数量
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(1,divnum+1):
		for j in range(len(scores)):
			if scores[j][i-1]>threshold:
				if target[j]==i:
					tp=tp+1
				else:
					fp=fp+1
			else:
				if target[j]==i:
					fn=fn+1
				else:
					tn=tn+1
	return tp,tn,fp,fn

#单分类计算准确度
def one_accuracy_score(target,scores,threshold):#目标，结果，第i个对象是正确对象
	tp=0
	tn=0
	fp=0
	fn=0

	for j in range(len(scores)):
		# if result[j]==1:
		#,8,7,8,6
		if scores[j]>threshold:
			if target[j]==1:
				tp=tp+1
			else:
				fp=fp+1
		else:
			if target[j]==1:
				fn=fn+1
			else:
				tn=tn+1
	return tp,tn,fp,fn


def one_accuracy_result(target,result):#目标，结果，第i个对象是正确对象
	tp=0
	tn=0
	fp=0
	fn=0

	for j in range(len(result)):
		if int(result[j])==1:
			if target[j]==1:
				tp=tp+1
			else:
				fp=fp+1
		else:
			if target[j]==1:
				fn=fn+1
			else:
				tn=tn+1
	return tp,tn,fp,fn




#孪生网络计算准确度（这里是小于，sklearn的是大于）
def siamese_accuracy_score(target,scores,threshold):#目标，结果，第i个对象是正确对象
	tp=0
	tn=0
	fp=0
	fn=0

	for j in range(len(scores)):

		if scores[j]<threshold:
			if target[j]==1:
				tp=tp+1
			else:
				fp=fp+1
		else:
			if target[j]==1:
				fn=fn+1
			else:
				tn=tn+1
	return tp,tn,fp,fn
	
def tripletloss_accuracy_score(target,scores,threshold):#目标，结果，第i个对象是正确对象
	tp=0
	tn=0
	fp=0
	fn=0

	for j in range(len(scores)):

		if scores[j]<threshold:
			if target[j]==1:
				tp=tp+1
			else:
				fp=fp+1
		else:
			if target[j]==1:
				fn=fn+1
			else:
				tn=tn+1
	return tp,tn,fp,fn



def cal_siamese_eer(test_label,score,startindex=0.01,maxindex=3,addindex=0.005):
	i=startindex
	while i<maxindex:
		tp,tn,fp,fn=siamese_accuracy_score(test_label,score,i)
		# print(tp,tn,fp,fn)
		accuracy=(tp+tn)/(tp+tn+fp+fn)
		far=(fp)/(fp+tn)
		frr=(fn)/(fn+tp)
		if frr<far or abs(frr-far)<0.01:
			break
		i=i+addindex
	print("i=",i)
	print("accuracy:",accuracy,"far:",far,"frr:",frr)
	return accuracy,far,frr