
# -*- coding=utf-8 -*-
import numpy as np
import os
import IAtool 
import filecontrol 
import MAfind
import featurecontrol
from normal_tool import *

# 单文件测试

filepath='./testdata/2020-09-06-10-07-41.csv'
print(filepath)
ppgx,ppgy,accx,accy,accz,gyrx,gyry,gyrz,ppgtime,acctime,gyrtime=filecontrol.orisegmentread(filepath)


orippgx=meanfilt(ppgx,20)
orippgy=meanfilt(ppgy,20)

butterppgx=highpass(5,200,orippgx)
butterppgy=highpass(5,200,orippgy)

# butterppgx=lowpass(1,200,orippgx)
# butterppgy=lowpass(1,200,orippgy)




icappgx,icappgy=IAtool.ppgfica(butterppgx,butterppgy)


tag,pointstartindex,pointendindex=MAfind.fine_grained_segment(icappgx,200,0.03)#python 的ica是0.03,android的是1

print(tag,pointstartindex,pointendindex,pointendindex-pointstartindex)
plt.plot(range(len(butterppgx)), butterppgx, 'blue')
plt.show()
plt.plot(range(len(icappgx)), icappgx, 'blue')
# plt.axvline(994,label='start point')#纵
# plt.axvline(1288,label='end point',color='grey')#纵
# plt.legend(loc ='upper right')
plt.show()

# # score=IAtool.short_time_energy(butterppgx)
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
# tempppgx=minmaxscale(orippgx)
# indexpicshow(orippgx)
# tag=MAfind.coarse_grained_detect(tempppgx)
# print(tag)