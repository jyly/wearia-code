# -*- coding=utf-8 -*-
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from minepy import MINE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

def indexpicshow(data):
	plt.plot(range(len(data)), data, 'blue')
	plt.show()


def stdpro(train_data,test_data):
    scaler = StandardScaler()
    scaler = scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    # print(np.dot(test_data[0]-lda.xbar_,lda.scalings_))
    #降维等于 np.dot(test_data-lda_bar,lda_scaling)
    if len(test_data)>0:
        test_data=scaler.transform(test_data)   
    # print(test_data[0])
    scaler_mean=[i for i in scaler.mean_]
    print("scaler_mean:",scaler_mean)
    scaler_scale=[i for i in scaler.scale_]
    print("scaler_scale:",scaler_scale)
    return train_data,test_data,scaler_mean,scaler_scale


def svm_accuracy_score(target,score,threshold,divnum):#目标，结果，目标对象的数量
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(0,divnum):
        for j in range(len(score)):
            if score[j][i]>threshold:
                if target[j]==i:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if target[j]==i:
                    fn=fn+1
                else:
                    tn=tn+1
    return tp,tn,fp,fn



def sklearn_svmclass(featureset,target,divnum):
    print(len(target))
    meanacc=[]
    meanfar=[]
    meanfrr=[]  
    for t in range(0,10):
        train_data,test_data, train_target, test_target = train_test_split(featureset,target,test_size = 0.2,random_state = t*30,stratify=target)
        
        train_data,test_data,scale_mean,scale_scale=stdpro(train_data,test_data)
        print("进入第",t,"轮分类阶段")
        clf = SVC(probability=True)

        clf.fit(X=train_data, y=train_target)

        result = clf.predict(test_data)
        score = clf.predict_proba(test_data)
        print('原结果：',test_target)
        print('预测结果：',result)
        print('预测分数：',score)
        # tp,tn,fp,fn=svm_accuracy_result(test_target,result,divnum)
        # accuracy=(tp+tn)/(tp+tn+fp+fn)
        # far=(fp)/(fp+tn)
        # frr=(fn)/(fn+tp)
        i=0.001
        far=1
        frr=0
        while far>frr:
            tp,tn,fp,fn=svm_accuracy_score(test_target,score,i,38)
            accuracy=(tp+tn)/(tp+tn+fp+fn)
            far=(fp)/(fp+tn)
            frr=(fn)/(fn+tp)
            # print("i=",i)
            # print("accuracy:",accuracy,"far:",far,"frr:",frr)
      
            i=i+0.001
        print("i=",i)
        print(tp,tn,fp,fn)
        accuracy=(tp+tn)/(tp+tn+fp+fn)
        far=(fp)/(fp+tn)
        frr=(fn)/(fn+tp)
        print("accuracy:",accuracy,"far:",far,"frr:",frr)

        meanacc.append(accuracy)
        meanfar.append(far)
        meanfrr.append(frr)
    print("meanacc:",np.mean(meanacc),"meanfar:",np.mean(meanfar),"meanfrr:",np.mean(meanfrr))
    for i in range(len(meanacc)):
        print("acc:",meanacc[i],"far:",meanfar[i],"frr:",meanfrr[i])





#计算单元积分
def callinelen(data):
	linelens=[]
	for i in range(len(data)-1):
		li=np.sqrt(1+(data[i+1]-data[i])**2)
		linelens.append(li)
	linelens.append(0)
	return linelens
#计算单元面积	
def callarea(data):
	areas=[]
	for i in range(len(data)-1):
		ar=float(data[i+1]+data[i])/2
		areas.append(ar)
	areas.append(0)
	return areas
# 计算单元微分
def calderivative(data):
	derivative=[]
	for i in range(len(data)-1):
		der=float(data[i+1]-data[i])
		derivative.append(der)
	derivative.append(0)
	return derivative	

def userfeature(username):
	dirpath="./fiducial_points/"+username
	filespace=os.listdir(dirpath)
	alldn=[]
	allsp=[]
	alldp=[]
	allsf=[]
	allef=[]
	for file in filespace:
		tempdn=[]
		tempsp=[]
		tempdp=[]
		tempsf=[]
		tempef=[]
		if file[-1]=='f':
			continue	
		files=dirpath+"/"+file
		print(files)

		jsons = json.load(open(files, "r"))
		# print(jsons)
		# print(len(jsons))

		for seg in range(len(jsons)):
			dn=jsons[str(seg)]['dychrotic_notch_i']
			sp=jsons[str(seg)]['systolic_peak_i']
			dp=jsons[str(seg)]['diastolic_peak_i']
			sf=jsons[str(seg)]['startPoint']
			ef=jsons[str(seg)]['endPoint']
			# print(dn,sp,dp,sf,ef)
			tempdn.append(dn)
			tempsp.append(sp)
			tempdp.append(dp)
			tempsf.append(sf)
			tempef.append(ef)
		alldn.append(tempdn)
		allsp.append(tempsp)
		alldp.append(tempdp)
		allsf.append(tempsf)
		allef.append(tempef)	

	alldata=[]
	dirpath="./beat/"+username
	for file in filespace:	
		tempdata=[]
		files=dirpath+"/"+file
		if file[-1]=='f':
			continue	
		print(files)
		jsons = json.load(open(files, "r"))
		keys = sorted(map(int, jsons["hb_argrelmin"].keys()))
		list_of_beats = [jsons["hb_argrelmin"][str(k)] for k in keys]
		# print(len(list_of_beats))
		# indexpicshow(list_of_beats[0])
		for seg in range(len(list_of_beats)):
			tempdata.append(list_of_beats[seg])
		alldata.append(tempdata)	
	print(len(alldata))		


	featureset=[]
	for ti in range(len(alldata)):
		if len(alldata[ti])<2:
			continue
		for seg in range(len(alldata[ti])-1):
			data=alldata[ti][seg]
			nextdata=alldata[ti][seg+1]

			lines=callinelen(data)
			nextlines=callinelen(nextdata)

			area=callarea(data)
			derivative=calderivative(data)


			dn=alldn[ti][seg]
			sp=allsp[ti][seg]
			dp=alldp[ti][seg]
			sf=allsf[ti][seg]
			ef=allef[ti][seg]

			nextdn=alldn[ti][seg+1]
			nextsp=allsp[ti][seg+1]
			nextdp=alldp[ti][seg+1]
			nextsf=allsf[ti][seg+1]
			nextef=allef[ti][seg+1]



			# yspsp=np.sum(lines[sp:ef])+np.sum(nextlines[nextsf:nextsp])
			yspsp=np.sqrt((nextdata[nextsp]-data[sp])**2+((nextsp-nextsf)+(ef-sp))**2)


			feature=[]

			# Point-Based
			feature.append(data[sp])
			feature.append(data[dn])
			feature.append(data[dp])

			feature.append(dp-sp)
			feature.append(nextdp-nextsf+ef-dp)#

			feature.append((data[sp]-data[sf])/(sp-sf))

			feature.append((sp-sf)/(ef-sf))
			feature.append((dn-sf)/(ef-sf))
			feature.append((dp-sf)/(ef-sf))
			feature.append((ef-dp)/(dp-sf))
			feature.append((dn-sp)/(ef-sp))#

			feature.append(abs(data[dn]-np.sum(lines[sf:dn])))
			# feature.append(abs(data[sf]-np.sum(lines[sp:ef])))#
			yspsf=np.sqrt((data[ef]-data[sp])**2+(ef-sp)**2)
			feature.append(abs(data[sf]-yspsf))#
			feature.append(abs(data[sp]-yspsp)/abs(data[sf]-yspsp))#

			#Area-Based
			feature.append(np.sum(area[sf:dn]))
			feature.append(np.sum(area[sf:dp]))
			feature.append(np.sum(area[dn:ef]))
			feature.append(np.sum(area[dp:ef]))

			#Statistic-Based
			count=0
			for i in range((ef-dp)):
				count=count+abs(derivative[dp+i])
			feature.append(count)	

			count=0
			for i in range((ef-sf)):
				count=count+abs(derivative[sf+i])
			feature.append(count)	
			vcount1=0
			vcount2=0
			ccount1=0
			ccount2=0
			for i in range((ef-sf)):
				if(derivative[sf+i]>0):
					vcount1=vcount1+derivative[sf+i]
					ccount1=ccount1+1
				else:
					vcount2=vcount2-derivative[sf+i]
					ccount2=ccount2+1
			feature.append(vcount1/vcount2)	
			count1=0
			count2=0
			for i in range((ef-sp)):
				count1=count1+abs(derivative[sp+i])
				count2=count2+abs(data[sp+i])
			feature.append(count1/count2)	
			count1=0
			count2=0
			for i in range((sp-sf)):
				count1=count1+abs(derivative[sf+i])
			for i in range((ef-sf)):
				count2=count2+abs(data[sf+i])
			feature.append(count1/count2)	
			feature.append((vcount1*vcount2)/(ccount1*ccount2))




			temp1=count1/(sp-sf)
			feature.append(temp1)
			count=0
			for i in range((ef-sp)):
				count=count+abs(derivative[sp+i])
			temp2=count/(ef-sp)
			feature.append(temp2)
			feature.append(temp1*temp2)
			count2=0
			for i in range((ef-sf)):
				count2=count2+abs(derivative[sf+i])
			feature.append((count/count2)*((ef-sp)/(ef-sf)))



			# ysfsp=np.sum(lines[sf:sp])
			ysfsp=np.sqrt((data[sp]-data[sf])**2+(sp-sf)**2)
			count=0
			for i in range((ef-sp)):
				count=count+abs(data[sp+i]-ysfsp)
			feature.append(count)
			count=0
			for i in range((sp-sf)):
				count=count+abs(data[sf+i]-ysfsp)
			feature.append(count)




			# ysfsf=np.sum(lines[sf:ef])
			ysfsf=np.sqrt((data[ef]-data[sf])**2+(ef-sf)**2)
			count=0
			for i in range((ef-sp)):
				count=count+abs(data[sp+i]-ysfsf)
			feature.append(count)
			count=0
			for i in range((ef-dp)):
				count=count+abs(data[dp+i]-ysfsf)
			feature.append(count)
			count1=0
			count2=0
			for i in range((sp-sf)):
				count1=count1+abs(data[sf+i]-ysfsf)
			for i in range((ef-sp)):
				count2=count2+abs(data[sp+i]-ysfsf)
			feature.append(count1/count2)
			count2=0
			for i in range((ef-sf)):
				count2=count2+abs(data[sf+i]-ysfsf)
			feature.append(count1/count2)



			count=0
			for i in range((dp-sp)):
				count=count+abs(data[sp+i]-yspsp)
			feature.append(count)
			count=0
			for i in range((dp-dn)):
				count=count+abs(data[dn+i]-yspsp)
			feature.append(count)
			count1=0
			count2=0
			for i in range((dn-sp)):
				count1=count1+abs(data[sp+i]-yspsp)
			for i in range((ef-sp)):
				count2=count2+abs(data[sp+i]-yspsp)
			feature.append(count1/count2)
			

			count1=0
			count2=0
			for i in range((sp-sf)):
				count1=count1+abs(data[sf+i]-ysfsf)
			for i in range((ef-sp)):
				count2=count2+abs(data[sp+i]-yspsp)
			for i in range((nextsp-nextsf)):
				count2=count2+abs(nextdata[nextsf+i]-yspsp)
			feature.append(count1/count2)

			featureset.append(feature)

	print(len(featureset))
	print(len(featureset[0]))
	return featureset

def featurewrite(dataset,targetset):
	featurefilepath='./features.csv'
	outputfile=open(featurefilepath,'w+')
	for i in range(len(targetset)):
		outputfile.write(str(targetset[i]))
		outputfile.write(',')
		for j in range(len(dataset[i])):
			outputfile.write(str(dataset[i][j]))
			outputfile.write(',')
		outputfile.write('\n')
	outputfile.close()

#将文件中的特征读出
def featureread():
	featurefilepath='./features.csv'
	inputfile=open(featurefilepath,'r+')
	dataset=[]
	targetset=[]
	for i in inputfile:
		i=list(eval(i))
		dataset.append(i[1:])
		targetset.append(i[0])
	inputfile.close()	
	#获取特征对应的用户数	
	dataset=np.array(dataset)
	targetset=np.array(targetset)
	print(dataset[0])
	print(targetset[0])
	return dataset,targetset



if __name__ == "__main__":

	dirpath="./fiducial_points/"
	userset=os.listdir(dirpath)
	dataset=[]
	targetset=[]
	for userindex in range(len(userset)):
		featureset=userfeature(userset[userindex])
		temptarget=[(userindex+1) for i in range(len(featureset))]
		targetset=targetset+temptarget
		dataset=dataset+featureset
	print(dataset[0])	
	print(targetset[0])	
	featurewrite(dataset,targetset)
	# dataset,targetset=featureread()
	sklearn_svmclass(dataset,targetset,len(userset))