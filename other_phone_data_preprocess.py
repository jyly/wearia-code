# -*- coding=utf-8 -*-
import os
from normal_tool import *
import filecontrol
#玛丽国王大学的数据
def single_data(filepath):
	accx=[]
	accy=[]
	accz=[]

	temp=[]
	inputfile=open(filepath,'r+')
	flag=0
	for i in inputfile:
		if i[0]==',':
			continue
		i=list(eval(i))
		# print(i)
		accx.append(i[1])
		accy.append(i[2])
		accz.append(i[3])

		if len(accx)==150:
			single=[]
			single.append(accx)
			single.append(accy)
			single.append(accz)
			temp.append(single)
			accx=[]
			accy=[]
			accz=[]
	inputfile.close()
	# if len(accx)>150:
	# 	accx=accx[-300:]
	# 	accy=accy[-300:]
	# 	accz=accz[-300:]
	# 	gyrx=gyrx[-300:]
	# 	gyry=gyry[-300:]
	# 	gyrz=gyrz[-300:]
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

	return temp

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
			singletemp=single_data(filepath)
			for i in singletemp:
				dataset[objnum].append(i)
		print("第",(objnum+1),"个动作的样本数：",len(dataset[objnum]))
		objnum=objnum+1	
	filecontrol.phone_datawrite(dataset)