# -*- coding=utf-8 -*-
import numpy as np
import os

def filewrite(filename,obj):
	f = open(filename,'wb')
	for line in obj.chunks():
		f.write(line)
	f.close()

def ppgread(path):
	ppgx=[]
	ppgy=[]
	input_1=open(path,'r+')
	for i in input_1:
		i=list(eval(i))
		ppgx.append(i[0])
		ppgy.append(i[1])
	input_1.close()	
	return ppgx,ppgy