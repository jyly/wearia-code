# -*- coding=utf-8 -*-
import os
from normal_tool import *
import filecontrol
#抬手行为的数据
def single_data(filepath):
	accx=[]
	accy=[]
	accz=[]

	gyrx=[]
	gyry=[]
	gyrz=[]
	inputfile=open(filepath,'r+')
	flag=0
	for i in inputfile:
		i=list(eval(i))
		# print(i)
		accx.append(i[0])
		accy.append(i[1])
		accz.append(i[2])
		gyrx.append(i[3])
		gyry.append(i[4])
		gyrz.append(i[5])
		if i[0]==99.0:
			flag=flag+1
		if flag==1:
			flag=flag+1
		if flag>150:
			break
	inputfile.close()
	if len(accx)>150:
		accx=accx[-300:]
		accy=accy[-300:]
		accz=accz[-300:]
		gyrx=gyrx[-300:]
		gyry=gyry[-300:]
		gyrz=gyrz[-300:]
		# accx=accx[:150]
		# accy=accy[:150]
		# accz=accz[:150]
		# gyrx=gyrx[:150]
		# gyry=gyry[:150]
		# gyrz=gyrz[:150]
		# accx=standardscale(accx)
		# accy=standardscale(accy)
		# accz=standardscale(accz)

		# gyrx=standardscale(gyrx)
		# gyry=standardscale(gyry)
		# gyrz=standardscale(gyrz)

	return accx,accy,accz,gyrx,gyry,gyrz

def all_data(datadir):
	dataset=[]
	oridataspace=os.listdir(datadir)
	objnum=0
	for filedirs in oridataspace:	
		dataset.append([])
		filedir=datadir+str(filedirs)+'/'
		filespace=os.listdir(filedir)
		for file in filespace:	
			filepath=filedir+str(file)
			print(filepath)
			accx,accy,accz,gyrx,gyry,gyrz=single_data(filepath)
			if len(accx)==300:
				temp=[]

				temp.append(accx)
				temp.append(accy)
				temp.append(accz)

				temp.append(gyrx)
				temp.append(gyry)
				temp.append(gyrz)

				dataset[objnum].append(temp)
		print("第",(objnum+1),"个动作的样本数：",len(dataset[objnum]))
		objnum=objnum+1	
	filecontrol.phone_datawrite(dataset)