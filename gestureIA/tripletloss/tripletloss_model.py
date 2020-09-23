# -*- coding=utf-8 -*-


from sklearn.model_selection import train_test_split
from tripletloss.tripletloss_base import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from sklearn.utils import shuffle
from keras.utils.data_utils import Sequence

# class tripletSequence(Sequence):
 
# 	def __init__(self, x_set, y_set, batch_size):
# 		self.x, self.y = x_set, y_set
# 		self.batch_size = batch_size
 
# 	def __len__(self):
# 		return int(np.ceil(len(self.x) / float(self.batch_size)))
 
# 	def __getitem__(self, idx):
# 		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
# 		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
 
# 		return np.array([
# 			resize(imread(file_name), (200, 200))
# 			for file_name in batch_x]), np.array(batch_y)

# def generate_arrays_from_file(path):
#             while True:
#                 with open(path) as f:
#                     for line in f:
#                         # create numpy arrays of input data
#                         # and labels, from each line in the file
#                         x1, x2, y = process_line(line)
#                         yield ({'input_1': x1, 'input_2': x2}, {'output': y})

def tripletSequence_1(data,target,batch_size=1):
	while 1:  
		datalen=len(data[0])
		for i in range(0,datalen):
			anchor=data[0][i]
			positive=data[1][i]
			negative=data[2][i]
			tempdata=[anchor,positive,negative]
			target1=np.zeros((1,len(target[0][0])))
			target2=np.zeros((1,len(target[1][0])))
			temptarget=[target1,target2]
			yield(tempdata,temptarget)		



#随机选择样本
def tripletSequence_3(data,target,batch_size):
	datalen=len(data)
	print(datalen)
	print("迭代次数：",(int(datalen/batch_size)+1))

	while 1:  
		cnt = 0
		temptriplet=[]
		templabel=[]
	
		for i in range(0,datalen):
			temptriplet.append(data[i])
			templabel.append(target[i])
			cnt += 1
			if cnt==batch_size or i==(datalen-1):
				cnt = 0

				temptriplet=np.array(temptriplet)
				templabel=np.array(templabel)
				anchor = temptriplet[:, 0].reshape(-1, 2, 300, 1)
				positive = temptriplet[:, 1].reshape(-1, 2, 300, 1)
				negative = temptriplet[:, 2].reshape(-1, 2, 300, 1)
				tempdata=[anchor,positive,negative]

				le = LabelBinarizer()
				y_anchor = le.fit_transform(templabel[:, 0])
				y_positive = le.fit_transform(templabel[:, 1])
				y_negative = le.fit_transform(templabel[:, 2])
				traintarget = np.concatenate((y_anchor, y_positive, y_negative), -1)
				temptarget=[traintarget,traintarget]

				yield(tempdata,temptarget)		
				temptriplet=[]
				templabel=[]

# class DataGenerator(keras.utils.Sequence):
  
#  def __init__(self, datas, batch_size=1, shuffle=True):
#   self.batch_size = batch_size
#   self.datas = datas
#   self.indexes = np.arange(len(self.datas))
#   self.shuffle = shuffle
 
#  def __len__(self):
#   #计算每一个epoch的迭代次数
#   return math.ceil(len(self.datas) / float(self.batch_size))
 
#  def __getitem__(self, index):
#   #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
#   # 生成batch_size个索引
#   batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#   # 根据索引获取datas集合中的数据
#   batch_datas = [self.datas[k] for k in batch_indexs]
 
#   # 生成数据
#   X, y = self.data_generation(batch_datas)
 
#   return X, y


def tripletloss_ori(train_data,test_data, train_target, test_target,num_classes):
	print(train_data.shape)

	input_shape = (2,300)
	train_data_flat = train_data.reshape(-1, 2*300)
	train_triplet, train_label = generate_triplet(train_data_flat, train_target, ap_pairs=20, an_pairs=100)
	print(train_triplet.shape)
	le = LabelBinarizer()

	model=create_tripletloss_network(input_shape,num_classes)

	train_triplet, train_label = shuffle(train_triplet, train_label, random_state=0)
	
	# train_triplet,val_triplet, train_label, val_label = train_test_split(train_triplet,train_label,test_size = 0.2,random_state = 30,stratify=train_label)


	anchor = train_triplet[:, 0].reshape(-1, 2, 300, 1)
	positive = train_triplet[:, 1].reshape(-1, 2, 300, 1)
	negative = train_triplet[:, 2].reshape(-1, 2, 300, 1)
	y_anchor = le.fit_transform(train_label[:, 0])
	y_positive = le.fit_transform(train_label[:, 1])
	y_negative = le.fit_transform(train_label[:, 2])
	traintarget = np.concatenate((y_anchor, y_positive, y_negative), -1)

	# target1=np.zeros((len(train_label),num_classes*3))
	# target2=np.zeros((len(train_label),384))


	# valanchor = val_triplet[:, 0].reshape(-1, 2, 300, 1)
	# valpositive = val_triplet[:, 1].reshape(-1, 2, 300, 1)
	# valnegative = val_triplet[:, 2].reshape(-1, 2, 300, 1)
	# val_anchor = le.fit_transform(val_label[:, 0])
	# val_positive = le.fit_transform(val_label[:, 1])
	# val_negative = le.fit_transform(val_label[:, 2])
	# valtarget = np.concatenate((val_anchor, val_positive, val_negative), -1)

	# target3=np.zeros((len(val_label),num_classes*3))
	# target4=np.zeros((len(val_label),384))




	model.fit(x=[anchor, positive, negative], y=[traintarget, traintarget],
          batch_size=1024, epochs=50, 
          validation_split=0.2)

	# model.fit([anchor, positive, negative], y=[traintarget, traintarget],
 #      batch_size=512, epochs=50, 
 #      validation_data=([valanchor, valpositive, valnegative], [valtarget, valtarget]))




	# batch_size=1024

	# model.fit_generator(tripletSequence_3(train_triplet, train_label, batch_size),
	# 	epochs=50,steps_per_epoch=(int(len(train_triplet)/batch_size)+1)
	# 	)

	# model.fit_generator(tripletSequence_3(train_triplet, target, batch_size),
	# 	epochs=50,steps_per_epoch=(int(len(train_triplet)/batch_size)+1),
	# 	validation_data=([valanchor, valpositive, valnegative], [valtarget, valtarget])
	# 	)


	model.save("triplet_loss_model.h5")

	base_network = mlp(input_shape,num_classes)
	input_a = Input(shape=input_shape, name='input_a')
	soft_a, pre_logits_a = base_network([input_a])
	model = Model(inputs=[input_a], outputs=[soft_a, pre_logits_a])
	model.load_weights("triplet_loss_model.h5")


	test_pairs, test_label = create_pairs(test_data, test_target,num_classes)
	test_soft_1, test_embed_1 = model.predict([test_pairs[:,0]])
	test_soft_2, test_embed_2 = model.predict([test_pairs[:,1]])

	score=[]
	for i in range(len(test_label)):
		score.append(K.sqrt(K.sum(K.square(test_embed_1[i] - test_embed_2[i]))))
	return score,test_label


def tripletloss_feature(train_data,test_data, train_target, test_target,num_classes):
	le = LabelBinarizer()

	input_shape = (len(train_data[0]))
	train_triplet, train_label = generate_triplet(train_data, train_target, ap_pairs=50, an_pairs=50)
	print("训练集组数：",train_triplet.shape)
	

	model=create_tripletloss_network(input_shape,num_classes)

	train_triplet, train_label = shuffle(train_triplet, train_label, random_state=0)
	
	# train_triplet,val_triplet, train_label, val_label = train_test_split(train_triplet,train_label,test_size = 0.2,random_state = 30,stratify=train_label)

	anchor = train_triplet[:, 0]
	positive = train_triplet[:, 1]
	negative = train_triplet[:, 2]
	y_anchor = le.fit_transform(train_label[:, 0])
	y_positive = le.fit_transform(train_label[:, 1])
	y_negative = le.fit_transform(train_label[:, 2])
	traintarget = np.concatenate((y_anchor, y_positive, y_negative), -1)

	# target1=np.zeros((len(train_label),num_classes*3))
	# target2=np.zeros((len(train_label),384))


	# valanchor = val_triplet[:, 0].reshape(-1, 2, 300, 1)
	# valpositive = val_triplet[:, 1].reshape(-1, 2, 300, 1)
	# valnegative = val_triplet[:, 2].reshape(-1, 2, 300, 1)
	# val_anchor = le.fit_transform(val_label[:, 0])
	# val_positive = le.fit_transform(val_label[:, 1])
	# val_negative = le.fit_transform(val_label[:, 2])
	# valtarget = np.concatenate((val_anchor, val_positive, val_negative), -1)

	# target3=np.zeros((len(val_label),num_classes*3))
	# target4=np.zeros((len(val_label),384))




	model.fit(x=[anchor, positive, negative], y=[traintarget, traintarget],
          batch_size=8192, epochs=100, 
          validation_split=0.2)

	# model.fit([anchor, positive, negative], y=[traintarget, traintarget],
 #      batch_size=512, epochs=50, 
 #      validation_data=([valanchor, valpositive, valnegative], [valtarget, valtarget]))

	# batch_size=1024

	# model.fit_generator(tripletSequence_3(train_triplet, train_label, batch_size),
	# 	epochs=50,steps_per_epoch=(int(len(train_triplet)/batch_size)+1)
	# 	)

	# model.fit_generator(tripletSequence_3(train_triplet, target, batch_size),
	# 	epochs=50,steps_per_epoch=(int(len(train_triplet)/batch_size)+1),
	# 	validation_data=([valanchor, valpositive, valnegative], [valtarget, valtarget])
	# 	)


	model.save("triplet_loss_model.h5")

	base_network = mlp_network_incre(input_shape,num_classes)
	input_a = Input(shape=input_shape, name='input_a')
	soft_a, pre_logits_a = base_network([input_a])
	model = Model(inputs=[input_a], outputs=[soft_a, pre_logits_a])
	model.load_weights("triplet_loss_model.h5")


	# test_pairs, test_label = create_pairs(test_data, test_target,num_classes)
	test_pairs, test_label = create_test_pair(test_data, test_target,num_classes)
	test_soft_1, test_embed_1 = model.predict([test_pairs[:,0]])
	test_soft_2, test_embed_2 = model.predict([test_pairs[:,1]])

	test_pred=[]
	for i in range(len(test_pairs)):
		test_pred.append(K.sqrt(K.sum(K.square(test_embed_1[i] - test_embed_2[i]))))
	
	print("len(test_pred):",len(test_pred))
	print("len(test_label):",len(test_label))
    #对样本对进行额外处理
	temp_pred=[]
	for i in range(int(len(test_label))):
		temppred=0
        # temp_label.append(test_label[i*5])
		for j in range(5):
			temppred+=test_pred[i*5+j]
		temp_pred.append(temppred/5)
	test_pred=temp_pred
	print("len(test_pred):",len(test_pred))
	print("len(test_label):",len(test_label))


	return test_pred,test_label


def tripletloss_feature_buildmodel(train_data, train_target,num_classes):
	le = LabelBinarizer()

	input_shape = (len(train_data[0]))
	train_triplet, train_label = generate_triplet(train_data, train_target, ap_pairs=50, an_pairs=50)
	print("训练集组数：",train_triplet.shape)

	model=create_tripletloss_network(input_shape,num_classes)

	train_triplet, train_label = shuffle(train_triplet, train_label, random_state=0)
	

	anchor = train_triplet[:, 0]
	positive = train_triplet[:, 1]
	negative = train_triplet[:, 2]
	y_anchor = le.fit_transform(train_label[:, 0])
	y_positive = le.fit_transform(train_label[:, 1])
	y_negative = le.fit_transform(train_label[:, 2])
	# traintarget = np.concatenate((y_anchor, y_positive, y_negative), -1)
	target1=np.zeros((len(train_label),10*3))
	# target2=np.zeros((len(train_label),384))
	traintarget=target1

	model.fit(x=[anchor, positive, negative], y=[traintarget, traintarget],
          batch_size=8192, epochs=40)

	model.save("triplet_loss_model.h5")


def tripletloss_feature_final(test_data,test_target,num_classes,anchornum=5):
	input_shape = (len(test_data[0]))

	base_network = mlp_network_incre(input_shape,num_classes)
	input_a = Input(shape=input_shape, name='input_a')
	soft_a, pre_logits_a = base_network([input_a])
	model = Model(inputs=[input_a], outputs=[soft_a, pre_logits_a])
	model.load_weights("triplet_loss_model.h5")


	# test_pairs, test_label = create_pairs(test_data, test_target,num_classes)
	test_pairs, test_label = create_test_pair(test_data, test_target,num_classes,anchornum)
	test_soft_1, test_embed_1 = model.predict([test_pairs[:,0]])
	test_soft_2, test_embed_2 = model.predict([test_pairs[:,1]])

	test_pred=[]
	for i in range(len(test_pairs)):
		test_pred.append(K.sqrt(K.sum(K.square(test_embed_1[i] - test_embed_2[i]))))

	# print("len(test_pred):",len(test_pred))
	# print("len(test_label):",len(test_label))
	#对样本对进行额外处理
	temp_pred=[]
	for i in range(int(len(test_label))):
		temppred=0
	    # temp_label.append(test_label[i*anchornum])
		for j in range(anchornum):
			temppred+=test_pred[i*anchornum+j]
		temp_pred.append(temppred/anchornum)
	test_pred=temp_pred
	print("len(test_pred):",len(test_pred))
	print("len(test_label):",len(test_label))
	test_pred=[K.eval(i) for i in test_pred]

	return test_pred,test_label

