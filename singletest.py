
# -*- coding=utf-8 -*-
import numpy as np
import os
import IAtool 
import filecontrol 
import MAfind
import featurecontrol
from normal_tool import *
import classifiercontrol

# 单文件测试

# ./segmentdata/8/2020-06-17-12-23-23.csv
# ./segmentdata/3/2020-06-16-16-15-30.csv
# ./segmentdata/3/2020-06-16-16-16-27.csv
# ./segmentdata/静止状态/2020-06-28-23-00-31.csv


# filepath='./segmentdata/静止状态/2020-06-28-23-00-31.csv'
filepath='./testdata/2020-07-01-16-01-04.csv'
print(filepath)
ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime=filecontrol.orisegmentread(filepath)
# ppgx,ppgy=filecontrol.ppgread(filepath)



orippgx=meanfilt(ppgx,20)
orippgy=meanfilt(ppgy,20)

butterppgx=highpass(2,200,orippgx)
butterppgy=highpass(2,200,orippgy)
# score=IAtool.short_time_energy(butterppgx)
# tempppgx=minmaxscale(butterppgx)
# score=IAtool.energy(tempppgx)
# score=IAtool.short_time_energy(butterppgx)

# plt.rc('font',family='Times New Roman') 
# plt.subplot(211)
# # plt.plot(range(len(butterppgx[120:-120])), butterppgx[120:-120], 'red',label='ppg data')
# plt.plot(range(len(butterppgx[1000:-200])), butterppgx[1000:-200], 'red',label='ppg data')

# plt.ylabel('PPG Reading') 
# plt.axvline(994,label='start point')#纵
# plt.axvline(1288,label='end point',color='grey')#纵
# plt.legend(loc ='upper right')

# plt.subplot(212)
# plt.plot(range(len(score[900:-300])), score[900:-300], 'blue' ,label='energy')

# plt.ylabel('Short Time Energy') 
# plt.xlabel('Sample index')
# # plt.axvline(284,label='undetected real start point')#纵
# # plt.axvline(727,color='grey',linestyle="-." ,label='undetected error start point')#纵
# plt.legend(loc ='upper right')
# plt.show()
tempppgx=minmaxscale(orippgx)
indexpicshow(orippgx)
tag=MAfind.coarse_grained_detect(tempppgx)
print(tag)