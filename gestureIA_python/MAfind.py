# -*- coding=utf-8 -*-
import numpy as np
import IAtool
from normal_tool import *
#提取自选间断数据的开始结束点
def fine_grained_segment(dn,fre,threshold=1):
	oristd=[]
	winslen=200
	for i in range(len(dn)-winslen):
		ori=np.std(dn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)

	pointstartindex=0
	pointendindex=0
	tag=0
	i=len(oristd)
	lens=int(0.8*fre)
	while i >lens:
		i=i-1
		# print(i)
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>threshold:	
			flag=0
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(i-lens,i):
				if oristd[j]<threshold:
					flag=1
					break
			if 0==flag:
				start=i-3*lens
				if start<0:
					start=0 
				for j in range(start,i-lens):
					pointstartindex=j
					if oristd[j]>threshold:
						break
				pointstartindex=pointstartindex+int(0.5*fre)
				pointendindex=i+int(0.5*fre)
				tag=1
				break
	if (pointendindex-pointstartindex)<150:
		tag=0
	return tag,pointstartindex,pointendindex



def fine_grained_segment_2(dn,fre,threshold=1):
	oristd=[]
	winslen=200
	for i in range(len(dn)-winslen):
		ori=np.std(dn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)
	pointstartindex=0
	pointendindex=0
	tag=0
	i=len(oristd)-fre - 350
	lens=int(fre)
	while i >(lens+50):
		i=i-1
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>threshold:	
			flag=0
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(0,lens):
				if oristd[i+j]<threshold:
					flag=1
					break
			if 0==flag:
				for j in range(0,lens):
					if oristd[i-j]>threshold+0.1:
						flag=1
						break
			if 0==flag:
				pointstartindex=i-100
				pointendindex=i+200
				tag=1
	return tag,pointstartindex,pointendindex



def fine_grained_segment_2(dn,fre,threshold=1):
	oristd=[]
	winslen=200
	for i in range(len(dn)-winslen):
		ori=np.std(dn[i:i+winslen])
		oristd.append(ori)
	# IAtool.indexpicshow(oristd)
	pointstartindex=0
	pointendindex=0
	tag=0
	i=len(oristd)-fre - 350
	lens=int(fre)
	while i >(lens+50):
		i=i-1
		#从后往前判断，当大于阈值时，认为可能存在手势，阈值根据经验判断，不同的滤波器的波动变化不同
		if oristd[i]>threshold:	
			flag=0
			#从后往前的一定区间内的值都大于阈值时，认为存在手势
			for j in range(0,lens):
				if oristd[i+j]<threshold:
					flag=1
					break
			if 0==flag:
				for j in range(0,lens):
					if oristd[i-j]>threshold+0.1:
						flag=1
						break
			if 0==flag:
				pointstartindex=i-50
				pointendindex=i+150
				tag=1	
	return tag,pointstartindex,pointendindex


def coarse_grained_detect(ppg,threshold=1):
	# indexpicshow(ppg)

	ppginter=IAtool.interationcal(ppg)
	# print(ppginter)
	# ppginter=[round(i,6) for i in ppg]
	# ppginter=meanfilt(ppginter,20)

	# ppginter=minmaxscale(ppginter)
	ppginter=standardscale(ppginter)
	ppginter=[round(i,1) for i in ppginter]

	indexpicshow(ppginter)
	# print(ppginter)

	# score=IAtool.energy(ppginter)
	# indexpicshow(score)

	alltag=IAtool.tagcal(ppginter)

	JS=[]
	for i in range(0,len(ppginter)-400,30):
		score1=IAtool.array_distribute_cal(ppginter[i:i+200],alltag)
		score2=IAtool.array_distribute_cal(ppginter[i+200:i+400],alltag)
		# tempjs=JS_divergence(score1,score2)
		# tempjs=cos_distance(score1,score2)
		# tempjs=calc_corr(score1,score2)
		tempjs=jaccard_distance(ppginter[i:i+200],ppginter[i+200:i+400])
		JS.append(tempjs)
	# print(JS)	
	# ppginter=meanfilt(ppginter,40)
	# indexpicshow(ppg)

	pointpicshow(JS)
	tag=0
	for i in range(len(JS)-6):
		flagnum=0
		# if JS[i]>0.35:
		if JS[i]<0.5:
			for j in range(i,i+6):
				# if JS[j]>0.35:
				if JS[j]<0.5:
					flagnum=flagnum+1
			if flagnum>4:
				tag=1
				break
	return tag