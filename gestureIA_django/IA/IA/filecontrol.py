# -*- coding=utf-8 -*-
import numpy as np
import os

def filewrite(filename,obj):
	f = open(filename,'wb')
	for line in obj.chunks():
		f.write(line)
	f.close()
