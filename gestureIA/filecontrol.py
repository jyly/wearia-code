# -*- coding=utf-8 -*-
import os
import numpy as np

#从文件中将所有数据按传感器进行分类
def orisegmentread(path):
	ppgx=[]
	ppgy=[]
	ppgtime=[]

	accx=[]
	accy=[]
	accz=[]
	acctime=[]

	gyrx=[]
	gyry=[]
	gyrz=[]
	gyrtime=[]

	inputfile=open(path,'r+')
	for i in inputfile:
		i=list(eval(i))
		if i[0]==0:
			accx.append(i[1])
			accy.append(i[2])
			accz.append(i[3])
			acctime.append(i[4])
		if i[0]==1:
			gyrx.append(i[1])
			gyry.append(i[2])
			gyrz.append(i[3])
			gyrtime.append(i[4])

		if i[0]==2:
			ppgx.append(i[1])
			ppgy.append(i[2])
			ppgtime.append(i[3])
	inputfile.close()	
	return ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime


#将采集特征按类别名存放到硬盘中
def featurewrite(feature,filename):
	dirpath='./selected_feature/'
	featurefilepath=dirpath+filename+'.csv'
	outputfile=open(featurefilepath,'w+')
	for i in range(len(feature)):
		for j in range(len(feature[i])):
			outputfile.write(str(feature[i][j]))
			outputfile.write(',')
		outputfile.write('\n')
	outputfile.close()

def featureread(dirpath='./selected_feature/'):
	filespace=os.listdir(dirpath)
	feature=[]
	target=[]
	index=1
	for file in filespace:	
		filepath=dirpath+str(file)
		print(filepath)
		inputfile=open(filepath,'r+')
		for i in inputfile:
			i=list(eval(i))
			feature.append(i)
			target.append(index)
		inputfile.close()	
		index=index+1
	#获取特征对应的用户数	
	targetnum=index-1
	return feature,target,targetnum


def datawrite(data,filesname):
	dirpath='./selected_madata/'
	datafilepath=dirpath+filesname+'.csv'
	outputfile=open(datafilepath,'w+')
	for i in range(len(data)):
		for t in range(8):#2代表仅录入ppg信号，8代表录入ppg信号和2个行为传感器信号
			for j in range(len(data[i][t])):#第index个用户的第i个样本的第t个信号的第j个信号点
				outputfile.write(str(data[i][t][j]))
				outputfile.write(',')
			outputfile.write('\n')
	outputfile.close()


def dataread():
	dirpath='./selected_madata/'
	filespace=os.listdir(dirpath)
	dataset=[]
	target=[]
	index=1
	for file in filespace:	
		filepath=dirpath+str(file)
		print(filepath)
		inputfile=open(filepath,'r+')
		temp=[]
		for i in inputfile:
			i=list(eval(i))
			if (len(i)!=300):
				print("有错误")
				continue
			if len(temp)<8:#2代表仅录入ppg信号，8代表录入ppg信号和2个行为传感器信号
				temp.append(i)
			else:
				dataset.append(temp)
				
				target.append(index)
				temp=[]
				temp.append(i)
		inputfile.close()	
		index=index+1
	targetnum=index-1
	dataset=np.array(dataset)
	target=np.array(target)
	return dataset,target,targetnum




def phone_datawrite(data):
	dirpath='./selected_madata/'
	for index in range(len(data)):
		datafilepath=dirpath+str(index+1)+'.csv'
		outputfile=open(datafilepath,'w+')
		for i in range(len(data[index])):
			for t in range(3):#6代表录入ppg信号和2个行为传感器信号
				for j in range(len(data[index][i][t])):#第index个用户的第i个样本的第t个信号的第j个信号点
					outputfile.write(str(data[index][i][t][j]))
					outputfile.write(',')
				outputfile.write('\n')
		outputfile.close()

def phone_dataread():
	dirpath='./selected_madata/'
	filespace=os.listdir(dirpath)
	dataset=[]

	target=[]
	index=1
	for file in filespace:	
		filepath=dirpath+str(file)
		print(filepath)
		inputfile=open(filepath,'r+')
		temp=[]
		for i in inputfile:
			i=list(eval(i))
			if (len(i)!=150):
				print("有错误")
				continue
			if len(temp)<3:#2代表2个行为传感器信号
				temp.append(i)
			else:
				dataset.append(temp)
				target.append(index)
				


				temp=[]
				temp.append(i)


		inputfile.close()	
		index=index+1
	

	targetnum=[]
	for i in range(len(target)):
		if target[i] not in targetnum:
			targetnum.append(target[i])
	targetnum=len(targetnum)
	print(targetnum)


	dataset=np.array(dataset)
	target=np.array(target)

	return dataset,target,targetnum