# -*- coding=utf-8 -*-
import os     



#读取原始数据，从文件中将所有数据按传感器进行分类
def orippgread(path):
	ppgx=[]
	ppgy=[]
	ppgtime=[]
	oldx=0
	oldy=0
	inputfile=open(path,'r+')
	for i in inputfile:
		i=list(eval(i))

		if i[0]==2:
			if(i[1]==0 or i[2]==0):
				continue
			if(len(ppg)<1):
				if(i[1]<100000000 and i[2]<100000000 and i[1]>50000 and i[2]>50000):
					continue
				else:
					oldx=i[1]
					oldy=i[2]
			x=abs(i[1]/oldx)
			y=abs(i[2]/oldy)
			if(x<10 and x>0.1 and y<10 and y>0.1):			

				ppgx.append(i[1])
				ppgy.append(i[2])
				ppgtime.append(i[3])
				oldx=i[1]
				oldy=i[2]
	inputfile.close()	
	return ppgx,ppgy

def JS_incretempdata(data,incres):
	tempdata=[]
	for i in range(incres):
		tempdata.append(data[i])
	for i in data:
		tempdata.append(i)
	for i in range(incres):
		tempdata.append(data[-1-i])
	return tempdata

def calenergy(data):
	tempdata=JS_incretempdata(data,100)
	energy=[]
	for i in range(len(data)):
		energy.append(np.mean(tempdata[i:i+200])+3*np.std(tempdata[i:i+200]))
		# energy.append(np.std(tempdata[i:i+200]))
	return energy

def bandpass(start,end,fre,data,order=3):#只保留start到end之间的频率的信号，fre是采样频率,order是滤波器阶数
	wa = start / (fre / 2) 
	we = end / (fre / 2) 
	b, a = signal.butter(order, [wa,we], 'bandpass')
	data = signal.filtfilt(b, a, data)
	return data



def coarse_grained_detect(ppg,threshold=1):
	# indexpicshow(ppg)
	# ppg=minmaxscale(ppg)

	ppg=meanfilt(ppg,10)

	ppginter=interationcal(ppg)
	# print(ppginter)
	# ppginter=[round(i,6) for i in ppg]
	# ppginter=meanfilt(ppginter,20)

	ppginter=minmaxscale(ppginter)
	# ppginter=standardscale(ppginter)
	# ppginter=minmaxscale(ppg)
	# ppginter=standardscale(ppg)
	orippginter=[round(i,1) for i in ppginter]

	# indexpicshow(ppginter)
	# print(ppginter)

	# score=IAtool.energy(ppginter)
	# indexpicshow(score)

	alltag=tagcal(orippginter)
	ppginter=JS_incretempdata(orippginter,200)
	JS=[]
	for i in range(0,len(ppginter)-400):

		score1=array_distribute_cal(ppginter[i:i+200],alltag)
		score2=array_distribute_cal(ppginter[i+200:i+400],alltag)
	
		tempjs=JS_divergence(score1,score2)
		# tempjs=cos_distance(score1,score2)
		if(i==30):
			print(score1)
			print(score2)
		# tempjs=calc_corr(score1,score2)
		# tempjs=jaccard_distance(ppginter[i:i+200],ppginter[i+200:i+400])
		JS.append(tempjs)
	# print(JS)	
	# ppginter=meanfilt(ppginter,40)
	# indexpicshow(ppg)

	# pointpicshow(JS)
	# mixindexpicshow(JS,orippginter)

	return JS




if __name__ == "__main__":
	gesturedir='./onlygesture'
	nogesturedir='./nogeture'

	dataspace=os.listdir(gesturedir)
	for filedirs in dataspace:
		dataset=[]
		filedir=gesturedir+str(fil)+'/'
		filespace=os.listdir(filedir)
		for file in filespaceL
			filepath=filedir+str(file)
			ppgx,ppgy=orippgread(filepath)
			
			butter=bandpass(2,5,200,ppgx)
			energy=calenergy(butter)