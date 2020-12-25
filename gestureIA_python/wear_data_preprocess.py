# -*- coding=utf-8 -*-
import os
import filecontrol 
import featurecontrol
from normal_tool import *
import IAtool 
import MAfind






def single_data(filepath):
	ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime=filecontrol.orisegmentread(filepath)
	
	# indexpicshow(ppgx)

	orippgx=meanfilt(ppgx,20)
	orippgy=meanfilt(ppgy,20)

	# oriaccx=meanfilt(accx,20)
	# oriaccy=meanfilt(accy,20)
	# oriaccz=meanfilt(accz,20)

	# origyrx=meanfilt(gyrx,20)
	# origyry=meanfilt(gyry,20)
	# origyrz=meanfilt(gyrz,20)

	#滤波滤除低频的信号
	butterppgx=highpass(2,200,orippgx)
	butterppgy=highpass(2,200,orippgy)

	#将手势数据和心跳数据分割开
	icappgx,icappgy=IAtool.ppgfica(butterppgx,butterppgy)

	#判断是否存在手势和提取出手势段
	# tag,pointstartindex,pointendindex=MAfind.fine_grained_segment(icappgx,200,0.03)#python 的ica是0.03,android的是1

	# tag,pointstartindex,pointendindex=MAfind.fine_grained_segment_2(icappgx,200,0.03)#python 的ica是0.03,android的是1
	tag,pointstartindex,pointendindex=MAfind.fine_grained_segment_4(icappgx,200,0.03)#python 的ica是0.03,android的是1

	#是否有手势，手势开始点，手势结束点，手势长度
	print(tag,pointstartindex,pointendindex,pointendindex-pointstartindex)
	if tag==1:
		# 将100hz的行为传感器数据拓展为200hz
		accx=IAtool.sequence_incre(accx)
		accy=IAtool.sequence_incre(accy)
		accz=IAtool.sequence_incre(accz)

		gyrx=IAtool.sequence_incre(gyrx)
		gyry=IAtool.sequence_incre(gyry)
		gyrz=IAtool.sequence_incre(gyrz)

		# accx=meanfilt(accx,20)
		# accy=meanfilt(accy,20)
		# accz=meanfilt(accz,20)

		# gyrx=meanfilt(gyrx,20)
		# gyry=meanfilt(gyry,20)
		# gyrz=meanfilt(gyrz,20)

		#数据标准化
		accx=standardscale(accx)
		accy=standardscale(accy)
		accz=standardscale(accz)

		gyrx=standardscale(gyrx)
		gyry=standardscale(gyry)
		gyrz=standardscale(gyrz)

		ppgx=standardscale(ppgx)
		ppgy=standardscale(ppgy)

		if (pointstartindex+350)<len(ppgx):
			ppgx=ppgx[pointstartindex:pointstartindex+300]
			ppgy=ppgy[pointstartindex:pointstartindex+300]
			
			accx=accx[pointstartindex:pointstartindex+300]
			accy=accy[pointstartindex:pointstartindex+300]
			accz=accz[pointstartindex:pointstartindex+300]

			gyrx=gyrx[pointstartindex:pointstartindex+300]
			gyry=gyry[pointstartindex:pointstartindex+300]
			gyrz=gyrz[pointstartindex:pointstartindex+300]
		else:
			tag=0

		# ppgx=IAtool.sequence_to_300(ppgx[pointstartindex:pointendindex])
		# ppgy=IAtool.sequence_to_300(ppgy[pointstartindex:pointendindex])
		
		# accx=IAtool.sequence_to_300(accx[pointstartindex:pointendindex])
		# accy=IAtool.sequence_to_300(accy[pointstartindex:pointendindex])
		# accz=IAtool.sequence_to_300(accz[pointstartindex:pointendindex])

		# gyrx=IAtool.sequence_to_300(gyrx[pointstartindex:pointendindex])
		# gyry=IAtool.sequence_to_300(gyry[pointstartindex:pointendindex])
		# gyrz=IAtool.sequence_to_300(gyrz[pointstartindex:pointendindex])
	return tag,ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz


#提取手势的具体数据段
def all_data(datadir):
	oridataspace=os.listdir(datadir)
	objnum=0
	for filedirs in oridataspace:	
		dataset=[]
		filedir=datadir+str(filedirs)+'/'
		filespace=os.listdir(filedir)
		for file in filespace:	
			filepath=filedir+str(file)
			print(filepath)
			tag,ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz=single_data(filepath)
			if tag==1:
				temp=[]
				temp.append(ppgx)
				temp.append(ppgy)

				temp.append(accx)
				temp.append(accy)
				temp.append(accz)

				temp.append(gyrx)
				temp.append(gyry)
				temp.append(gyrz)

				dataset.append(temp)
		print("第",(objnum+1),"个手势的样本数：",len(dataset))
		objnum=objnum+1	
		filecontrol.datawrite(dataset,filedirs)





def single_feature(filepath):

	ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime=filecontrol.orisegmentread(filepath)
	if len(ppgx)<800:
		tag=0
		tempfeature=[]
		return tag,tempfeature

	orippgx=meanfilt(ppgx,20)
	orippgy=meanfilt(ppgy,20)

	# accx=meanfilt(accx,20)
	# accy=meanfilt(accy,20)
	# accz=meanfilt(accz,20)

	# gyrx=meanfilt(gyrx,20)
	# gyry=meanfilt(gyry,20)
	# gyrz=meanfilt(gyrz,20)

	butterppgx=highpass(2,200,orippgx)
	butterppgy=highpass(2,200,orippgy)

	icappgx,icappgy=IAtool.ppgfica(butterppgx,butterppgy)
	icappgx,icappgy=IAtool.ppgfica(icappgx,icappgy)
	
	# orippgx=minmaxscale(orippgx)
	# orippgx=standardscale(orippgx)
	# tag=MAfind.coarse_grained_detect(orippgx)
	# print(tag)
	tag,pointstartindex,pointendindex=MAfind.fine_grained_segment(icappgx,200,0.03)#0.03,1
	# tag,pointstartindex,pointendindex=MAfind.fine_grained_segment_2(icappgx,200,0.03)#python 的ica是0.03,android的是1
	# tag,pointstartindex,pointendindex=MAfind.fine_grained_segment_4(icappgx,200,0.03)#python 的ica是0.03,android的是1

	print(tag,pointstartindex,pointendindex)
	# mixindexpicshow(orippgx,icappgx)

	# orippgx=standardscale(orippgx)
	# orippgy=standardscale(orippgy)
	# accx=standardscale(accx)
	# accy=standardscale(accy)
	# accz=standardscale(accz)
	# gyrx=standardscale(gyrx)
	# gyry=standardscale(gyry)
	# gyrz=standardscale(gyrz)

	tempfeature=[]
	if tag==1:
		temp=featurecontrol.ppg_feature(orippgx[pointstartindex:pointendindex])
		for i in temp:
			tempfeature.append(i)
		temp=featurecontrol.ppg_feature(orippgy[pointstartindex:pointendindex])
		for i in temp:
			tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(accx[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(accy[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(accz[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(gyrx[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(gyry[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)

	# 	temp=featurecontrol.motion_feature(gyrz[int((pointstartindex)/2):int((pointendindex)/2)])
	# 	for i in temp:
	# 		tempfeature.append(i)
	return tag,tempfeature

# #提取手势具体数据段的特征
def all_feature(datadir):
	oridataspace=os.listdir(datadir)
	objnum=0
	featurenumset=[]
	for filedirs in oridataspace:	
		featureset=[]
		filedir=datadir+str(filedirs)+'/'
		filespace=os.listdir(filedir)
		for file in filespace:	
			filepath=filedir+str(file)
			print(filepath)
			tag,tempfeature=single_feature(filepath)
			if tag==1:
				featureset.append(tempfeature)
		print("第",(objnum+1),"个手势的样本数：",len(featureset))
		featurenumset.append(len(featureset))
		objnum=objnum+1
		filecontrol.featurewrite(featureset,filedirs)

	print("每个类别的样本数：",featurenumset)


def renew_feature():
	datadirpath='./selected/madata/'
	featuredirpath='./selected/feature/'
	filespace=os.listdir(datadirpath)
	for file in filespace:	
		dataset=[]
		filepath=datadirpath+str(file)
		print(filepath)
		inputfile=open(filepath,'r+')
		temp=[]
		for i in inputfile:
			i=list(eval(i))
			temp.append(i)
			if len(temp)==8:#2代表仅录入ppg信号，8代表录入ppg信号和2个行为传感器信号
				# if len(temp[0])>300:
				# 	temp=np.array(temp)
				# 	dataset.append(temp[:,:300])
				dataset.append(temp)
				temp=[]
		inputfile.close()
		feature=[]
		for i in range(len(dataset)):
			temp=[]
			# dataset[i]=IAtool.data_resize(dataset[i],200)

			temp=temp+featurecontrol.ppg_feature(dataset[i][0])
			temp=temp+featurecontrol.ppg_feature(dataset[i][1])
			# print(len(temp))
			# butter=bandpass(2,5,200,dataset[i][0])
			# temp=temp+featurecontrol.ppg_feature(butter)
			# butter=bandpass(2,5,200,dataset[i][1])
			# temp=temp+featurecontrol.ppg_feature(butter)


			temp=temp+featurecontrol.motion_feature(dataset[i][2])
			temp=temp+featurecontrol.motion_feature(dataset[i][3])
			temp=temp+featurecontrol.motion_feature(dataset[i][4])
			temp=temp+featurecontrol.motion_feature(dataset[i][5])
			temp=temp+featurecontrol.motion_feature(dataset[i][6])
			temp=temp+featurecontrol.motion_feature(dataset[i][7])
			# print(len(temp))
			feature.append(temp)

		featurefilepath=featuredirpath+str(file)
		print(featurefilepath)
		outputfile=open(featurefilepath,'w+')
		for i in range(len(feature)):
			for j in range(len(feature[i])):
				outputfile.write(str(feature[i][j]))
				outputfile.write(',')
			outputfile.write('\n')
		outputfile.close()