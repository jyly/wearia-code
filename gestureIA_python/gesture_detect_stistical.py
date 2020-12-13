# -*- coding=utf-8 -*-
import os
import filecontrol 
from normal_tool import *
import MAfind


#对有手势的数据进行手势判断
def detect_cal_true(datadir):
	oridataspace=os.listdir(datadir)
	objnum=0
	indexcal=0
	energycal=0
	shannoncal=0
	jscal=0
	for filedirs in oridataspace:	
		dataset=[]
		filedir=datadir+str(filedirs)+'/'
		filespace=os.listdir(filedir)
		for file in filespace:	
			filepath=filedir+str(file)
			print(filepath)
			ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime=filecontrol.oridataread(filepath)

			butter=bandpass(2,5,200,ppgx)
			energy=MAfind.calenergy(butter)

			# plt.subplot(3,1,1)
			# plt.plot(range(len(butter)), butter, 'red',linewidth=0.6)
			# plt.subplot(3,1,2)
			# plt.plot(range(len(energy)), energy, 'red',linewidth=0.6)
			# plt.subplot(3,1,3)
			# plt.plot(range(len(JS)), JS, 'red',linewidth=0.6)
			# plt.show()

			indexcal=indexcal+1
			for i in range(len(energy)):
				if energy[i]>3500:
					flagnum=0
					for j in range(i,i+150):
						if j==len(energy):
							break
						if energy[j]>3500:
							flagnum=flagnum+1
					if flagnum>100:
						energycal=energycal+1
						break

			butterround=minmaxscale(butter)
			butterround=[round(i,2) for i in butterround]
			butterround=np.array(butterround)
			butterround=butterround.reshape(-1,1)
			butterround=calcShannonEnt(butterround)
			if butterround<6:
				shannoncal=shannoncal+1	


			JS=MAfind.coarse_grained_detect(energy)
			for i in range(len(JS)):
				if JS[i]>0.2:
					flagnum=0
					for j in range(i,i+150):
						if j==len(JS):
							break
						if JS[j]>0.15:
							flagnum=flagnum+1
					if flagnum>130:
						jscal=jscal+1
						break


			print(indexcal,energycal,shannoncal,jscal)
		print(indexcal,energycal,shannoncal,jscal)

	print(indexcal,energycal,shannoncal,jscal)

def detect_cal_false(datadir):
	oridataspace=os.listdir(datadir)
	objnum=0
	indexcal=0
	energycal=0
	shannoncal=0
	jscal=0
	for file in oridataspace:	
		filepath=datadir+str(file)
		print(filepath)
		dataset=[]

		inputfile=open(filepath,'r+')
		for i in inputfile:
			i=list(eval(i))
			dataset.append(i)
		inputfile.close()

		ppgx=dataset[0]
		ppgy=dataset[1]

		butter=bandpass(2,5,200,ppgx)
		energy=MAfind.calenergy(butter)
		indexcal=indexcal+1

		for i in range(len(energy)):
			if energy[i]>3500:
				flagnum=0
				for j in range(i,i+150):
					if j==len(energy):
						break
					if energy[j]>3500:
						flagnum=flagnum+1
				if flagnum>130:
					# plt.plot(range(len(butter)), butter, 'red',linewidth=0.6)
					# plt.show()	
					energycal=energycal+1
					break


		butterround=minmaxscale(butter)
		butterround=[round(i,2) for i in butterround]
		butterround=np.array(butterround)
		butterround=butterround.reshape(-1,1)
		butterround=calcShannonEnt(butterround)
		if butterround<6:
			shannoncal=shannoncal+1	

			# plt.plot(range(len(butter)), butter, 'red',linewidth=0.6)
			# plt.show()	


		JS=MAfind.coarse_grained_detect(energy)
		for i in range(len(JS)):
			if JS[i]>0.15:
				flagnum=0
				for j in range(i,i+150):
					if j==len(JS):
						break
					if JS[j]>0.15:
						flagnum=flagnum+1
				if flagnum>130:
					jscal=jscal+1
					# plt.plot(range(len(butter)), butter, 'red',linewidth=0.6)
					# plt.show()	
					break


		# plt.subplot(3,1,1)
		# plt.plot(range(len(butter)), butter, 'red',linewidth=0.6)
		# plt.subplot(3,1,2)
		# plt.plot(range(len(energy)), energy, 'red',linewidth=0.6)
		# plt.subplot(3,1,3)
		# plt.plot(range(len(JS)), JS, 'red',linewidth=0.6)
		# plt.show()
					
		print(indexcal,energycal,shannoncal,jscal)
	print(indexcal,energycal,shannoncal,jscal)


