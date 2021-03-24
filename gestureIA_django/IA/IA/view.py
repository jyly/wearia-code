from django.http import HttpResponse
from ast import literal_eval
from django.http import JsonResponse
import os
import time
from . import filecontrol
import matplotlib.pyplot as plt


dirpath=os.getcwd()+'/data/'
dirpath=dirpath.replace('\\','/')
def hello(request):
	print(dirpath)
	return HttpResponse("welecome to use the django service !")

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

		if os.path.exists(filename):
			os.remove(filename)

		filecontrol.filewrite(filename,obj)
		str['updata']=['success']

	else:
		str['updata']=['false']
		return HttpResponse("IA")
	return JsonResponse(str)