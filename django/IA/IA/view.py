from django.http import HttpResponse
from ast import literal_eval
from django.http import JsonResponse
import os
import time
from . import filecontrol
import matplotlib.pyplot as plt
# fig = plt.figure()
dirpath=os.getcwd()+'/data/'
dirpath=dirpath.replace('\\','/')
def hello(request):
	print(dirpath)
	return HttpResponse("Hello world ! ")

def IA(request):
	obj = request.FILES.get("data")
	str={}
	if obj:
		data=obj.name
		data=literal_eval(data)
		print(data)

		filepath=dirpath+data['sensor']+'/'
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		
		filepath=filepath+data['username']+'/'
		if not os.path.exists(filepath):
			os.makedirs(filepath)

		filepath=filepath+data['username']+'_'+data['gesture_item']+'/'
		if not os.path.exists(filepath):
			os.makedirs(filepath)

		downtime=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

		filename=filepath+downtime+'.csv'
		print(filename)

		# if (data['sensor']=='ppg'):
		# 	filename=filepath+downtime+'.csv'
		# 	print(filename)
		# if (data['sensor']=='rawppg'):
		# 	filename=filepath+'raw-'+downtime+'.csv'
		# 	print(filename)
		# if (data['sensor']=='motion'):
		# 	filename=filepath+'motion-'+downtime+'.csv'
		# 	print(filename)
		# if (data['sensor']=='icappg'):
		# 	filename=filepath+'ica-'+downtime+'.csv'
		# if (data['sensor']=='feature'):
		# 	filename=filepath+'feature-'+downtime+'.csv'

		# if (data['sensor']=='feature'):
		# 	filename=filepath+'feature-'+downtime+'.csv'

		if os.path.exists(filename):
			os.remove(filename)

		filecontrol.filewrite(filename,obj)
		# plt.close()

		# ppgx,ppgy=filecontrol.ppgread(filename)

		# fig = plt.figure()
		# plt.subplot(311)
		# plt.plot(range(len(ppgx)), ppgx, 'red')
		# plt.subplot(312)
		# plt.plot(range(len(ppgy)), ppgy, 'blue')
		# plt.subplot(313)
		# plt.plot(range(len(ppgx)), ppgx, 'red')
		# plt.plot(range(len(ppgy)), ppgy, 'blue')
		# # plt.show()
		# filename='E:/pic/raw-'+downtime+'.png'
		# fig.savefig(filename)
		# plt.close()

		str['updata']=['success']



		# if (data['sensor']=='feature'):
		# 	mineindex=[32,39,33,38,34,36,37,35,11,10,68,50,23,22,2,1,51,45,49,19]
		# 	ldamatrix=[[ 3.41568494e+03], [-7.78971311e+02], [ 1.02197192e+02], [ 3.75869543e+03], [-3.75765998e+03], 
		# 	[ 7.39264809e+02], [-5.23010843e+03], [ 4.13574600e+03], [-8.57698454e+05], [ 8.52935746e+05], 
		# 	[ 6.63736708e+02], [-6.30335503e+04], [ 1.92467789e+04], [-1.85630784e+04], [ 4.28034513e+04], 
		# 	[-4.49471795e+04], [ 6.07281691e+04], [-8.01463460e+00], [-8.01463460e+00], [ 2.92652350e+03]]

	else:
		str['updata']=['false']
		return HttpResponse("IA")
	return JsonResponse(str)