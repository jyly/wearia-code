# -*- coding=utf-8 -*-
from sklearn.model_selection import train_test_split
from siamese.siamese_base import *
from siamese.siamese_tools import *
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import os

def siamese_authentication(train_data,test_data, train_target,test_target,trainindex,testindex,anchornum):
    train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
    test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)

    print("训练集对数：",train_pairs.shape)
    print("测试集对数：",test_pairs.shape)

    train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
    train_pairs,test_pairs,train_label,test_label=dataresize(train_pairs,test_pairs,train_label,test_label)

    input_shape = (len(test_pairs[0][0]),len(test_pairs[0][0][0]))
    print(input_shape)
    
    model,based_model=create_siamese_network(input_shape,'pyramid_lstm')
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
           batch_size=4086, epochs=20)  

    test_pred = model.predict([test_pairs[:,0], test_pairs[:,1]])
    # test_pred=repro_test_pred(test_pred,test_label,anchornum)
    test_pred=np.array(test_pred)
    return test_pred,test_label

def siamese_data_build(train_data, train_target,num_classes):
    train_pairs, train_label = create_pairs_incre(train_data,train_target,num_classes)
    print("训练集对数：",train_pairs.shape)
    train_pairs,train_label = shuffle(train_pairs, train_label, random_state=10)
    train_pairs,train_label=dataresize(train_pairs,train_label)
    input_shape = [len(train_pairs[0][0]),len(train_pairs[0][0][0])]
    print(input_shape)

    model,based_model=create_siamese_network(input_shape,'pyramid')
    history = model.fit([train_pairs[:,0], train_pairs[:,1]], train_label,  
           batch_size=4096, epochs=200)  

    model.save_weights(os.path.abspath('.').replace("\\",'/')+'/parameter/model_weights.h5')#原型因有lamdba层，不能直接保存模型
    based_model.save_weights(os.path.abspath('.').replace("\\",'/')+'/parameter/based_model_weights.h5')
    based_model.save(os.path.abspath('.').replace("\\",'/')+'/parameter/based_model.h5')
  

def siamese_data_test(test_data,test_target,num_classes,anchornum=5):
    test_pairs, test_label = create_pairs_incre(test_data, test_target,num_classes)
    print("测试集对数：",test_pairs.shape)
    test_pairs,test_label=dataresize(test_pairs,test_label)
    input_shape = [len(test_pairs[0][0]),len(test_pairs[0][0][0])]
    print(input_shape)
    #方案1求出文件中模型的预测结果
    model,based_model=create_siamese_network(input_shape,'pyramid')
    model.load_weights(os.path.abspath('.').replace("\\",'/')+'/parameter/model_weights.h5')
    test_pred = model.predict([test_pairs[:,0], test_pairs[:,1]])

    # 保存为tflite文件,给移动端使用
    # # based_model =load_model('./parameter/based_model.h5')
    # # based_model.load_weights('./parameter/based_model_weights.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model_file('./parameter/based_model.h5')
    # # converter = tf.lite.TFLiteConverter.from_keras_model(based_model)
    # tflite_model = converter.convert()
    # open("./parameter/based_model.tflite", "wb").write(tflite_model)
    # interpreter = tf.lite.Interpreter(model_path="./parameter/based_model.tflite")
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    test_label=np.array(test_label)
    test_pred=np.array(test_pred)
    return test_pred,test_label





#多任务网络构建
# def siamese_mul_task(train_data,test_data, train_target,test_target,trainindex,testindex,anchornum):
#     train_pairs, train_label = create_multask_pair(train_data, train_target,trainindex)
#     test_pairs, test_label = create_multask_pair(test_data, test_target,testindex)
#     print("训练集对数：",train_pairs.shape)
#     print("测试集对数：",test_pairs.shape)
#     train_user_pairs, train_user_label = shuffle(train_pairs[0], train_label[0], random_state=10)
#     train_gesture_pairs, train_gesture_label = shuffle(train_pairs[1], train_label[1], random_state=10)
#     train_pairs=[train_user_pairs,train_gesture_pairs]
#     train_label=[train_user_label,train_gesture_label]
#     train_pairs=np.array(train_pairs)
#     train_label=np.array(train_label)
#     train_pairs,test_pairs,train_label,test_label=dataresize(train_pairs,test_pairs,train_label,test_label)
#     input_shape = (len(train_user_pairs[0][0]),len(train_user_pairs[0][0][0]))
#     print(input_shape)
#     model,based_model=create_mul_task_siamese_network(input_shape)
#     history = model.fit([train_pairs[0][:, 0], train_pairs[0][:, 1],train_pairs[1][:, 0], train_pairs[1][:, 1]], 
#         [train_label[0],train_label[1]],  
#            batch_size=512, epochs=1)  
#     test_user_pred,test_gesture_pred = model.predict([test_pairs[0][:,0], test_pairs[0][:,1],test_pairs[1][:,0], test_pairs[1][:,1]])
#     test_pred=[test_user_pred,test_gesture_pred]
#     # test_pred=repro_test_pred(test_pred,test_label,anchornum)
#     test_pred=np.array(test_pred)
#     return test_pred,test_label


#
# def siamese_mul_model_data(train_data,test_data, train_target,test_target,trainindex,testindex,anchornum):
#     train_pairs, train_label = create_pairs_incre(train_data, train_target,trainindex)
#     test_pairs, test_label = create_pairs_incre(test_data, test_target,testindex)
#     print("训练集对数：",train_pairs.shape)
#     print("测试集对数：",test_pairs.shape)
#     #长-宽-高
#     train_pairs=train_pairs.reshape(len(train_pairs),2,len(train_data[0]),len(train_data[0][0]),1)
#     test_pairs=test_pairs.reshape(len(test_pairs),2,len(test_data[0]),len(test_data[0][0]),1)
#     print("训练集对数：",train_pairs.shape)
#     print("测试集对数：",test_pairs.shape)
#     train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
#     ppg_input_shape = (2,len(train_data[0][0]),1)
#     motion_input_shape = (6,len(train_data[0][0]),1)
#     model,base_network_1,base_network_2=create_mul_data_siamese_network(ppg_input_shape,motion_input_shape)
#     history = model.fit([train_pairs[:, 0,:2], train_pairs[:, 1,:2],train_pairs[:, 0,2:], train_pairs[:, 1,2:]]
#         ,train_label,batch_size=256, epochs=10,validation_split=0.2)  
#     test_pred = model.predict([test_pairs[:, 0,:2], test_pairs[:, 1,:2],test_pairs[:, 0,2:], test_pairs[:, 1,2:]])
#     test_pred=np.array(test_pred)
#     return test_pred,test_label











# def siamese_mul_data_buildmodel(train_data, train_target,num_classes,featurenum):
#     train_pairs, train_label = create_pairs_incre_1(train_data, digit_indices,num_classes)
#     # train_pairs, train_label = create_pairs_incre_2(train_data, train_target,num_classes)

#     print("train_pairs.shape:",train_pairs.shape)
     
#     input_shape = (len(train_data[0]),len(train_data[0][0]))
#     print(input_shape)

#     train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
#     ppg_model,ppg_based_model=create_siamese_network(input_shape)
#     history = ppg_model.fit([train_pairs[:, 0,:2], train_pairs[:, 1,:2]], train_label,  
#            batch_size=128, epochs=40)  
#     ppg_model.save_weights('./parameter/ppg_model_weights.h5')#原型因有lamdba层，不能直接保存模型
#     ppg_based_model.save_weights('./parameter/ppg_based_model_weights.h5')
#     ppg_based_model.save('./parameter/ppg_based_model.h5')
  
#     motion_model,motion_based_model=create_siamese_network(input_shape)
#     history = motion_model.fit([train_pairs[:, 0,2:], train_pairs[:, 1,2:]], train_label,  
#            batch_size=128, epochs=40)  
#     motion_model.save_weights('./parameter/motion_model_weights.h5')
#     motion_based_model.save_weights('./parameter/motion_based_model_weights.h5')
#     motion_based_model.save('./parameter/motion_based_model.h5')

# def siamese_mul_data_final(test_data,test_target,num_classes,featurenum,anchornum=5):
#     # test_pairs, test_label = create_pairs_incre_1(data, target,num_classes)
#     test_pairs, test_label = create_test_pair(data, target,num_classes,anchornum)
#     # test_pairs, test_label = create_victima_test_pair(data, target,num_classes,anchornum)
   
#     ppg_input_shape = (len(test_data[0]),len(test_data[0][0]))
#     print(ppg_input_shape)

#     ppg_model,ppg_based_network=create_siamese_network(ppg_input_shape)
#     ppg_model.load_weights('./parameter/ppg_model_weights.h5')
#     ppg_based_network =load_model('./parameter/ppg_based_model.h5')
#     converter = tf.lite.TFLiteConverter.from_keras_model(ppg_based_network)
#     tflite_model = converter.convert()
#     open("ppg_based_model.tflite", "wb").write(tflite_model)

#     motion_input_shape = (len(test_data[0]),len(test_data[0][0]))
#     print(motion_input_shape)
#     motion_model,motion_based_network=create_siamese_network(motion_input_shape)
#     motion_model.load_weights('motion_model_weights.h5')
#     motion_based_network =load_model('motion_based_model.h5')
#     converter = tf.lite.TFLiteConverter.from_keras_model(motion_based_network)
#     tflite_model = converter.convert()
#     open("motion_based_model.tflite", "wb").write(tflite_model)


#     ppg_test_pred = ppg_model.predict([test_pairs[:, 0,:2], test_pairs[:, 1,:2]])
#     motion_test_pred = motion_model.predict([test_pairs[:, 0,2:], test_pairs[:, 1,2:]])
 

#     #对样本对进行额外处理
#     temp_pred=[]
#     for i in range(int(len(test_label))):
#         temppred=0
#         for j in range(anchornum):
#             temppred+=ppg_test_pred[i*anchornum+j]
#         temp_pred.append(temppred/anchornum)
#     ppg_test_pred=temp_pred


#     temp_pred=[]
#     for i in range(int(len(test_label))):
#         temppred=0
#         # temp_label.append(test_label[i*5])
#         for j in range(anchornum):
#             temppred+=motion_test_pred[i*anchornum+j]
#         temp_pred.append(temppred/anchornum)
#     motion_test_pred=temp_pred

#     # print("len(ppg_test_pred):",len(ppg_test_pred))
#     # print("len(motion_test_pred):",len(motion_test_pred))


#     test_pred=[]
#     for i in range(len(ppg_test_pred)):
#         test_pred.append((ppg_test_pred[i]+motion_test_pred[i])/2)

#     test_pred=ppg_test_pred

#     test_label=np.array(test_label)
#     test_pred=np.array(test_pred)
#     return test_pred,test_label




# def siamese_cwt(train_data,test_data, train_target, test_target,num_classes):


#     # train_pairs, train_label = create_pairs(train_data, digit_indices,num_classes)
#     train_pairs, train_label = create_pairs_incre_1(train_data, train_target,num_classes)
#     # train_pairs, train_label = create_pairs_incre_2(train_data, digit_indices,num_classes)

#     # test_pairs, test_label = create_pairs(test_data, digit_indices,num_classes)
#     test_pairs, test_label = create_pairs_incre_1(test_data, test_target,num_classes)
#     # test_pairs, test_label = create_pairs_incre_2(test_data, digit_indices,num_classes)
    
#     print(train_pairs.shape)
#     print(test_pairs.shape)
    
#     input_shape = (len(train_data[0]),len(train_data[0][0]),2)
#     # input_shape = (2,24,300)
#     # train_pairs = train_pairs.reshape(train_pairs.shape[0], 2, 2, 64, 300)  #配对个数，每对有2组数据，每组数据有2个信号源，每个信号源有32维，300长
#     # test_pairs = test_pairs.reshape(test_pairs.shape[0], 2, 2, 64, 300) 
#     model=create_siamese_network(input_shape)
#     # model=create_presudo_siamese_network(input_shape)

#     #配对数，对子内部数据段个数，数据段的长，数据段的宽，数据段的高
 
#     train_pairs, train_label = shuffle(train_pairs, train_label, random_state=10)
#     train_pairs = train_pairs.astype(np.float16)
#     train_label = train_label.astype(np.float16)

#     history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,  
#            batch_size=128,epochs=50,
#            validation_split=0.2)  
    
#     # train_pairs,val_pairs, train_label, val_label = train_test_split(train_pairs,train_label,test_size = 0.2,random_state = 30,stratify=train_label)
#     # train_pairs = train_pairs.astype(np.float16)
#     # train_label = train_label.astype(np.float16)
#     # val_pairs = val_pairs.astype(np.float16)
#     # val_label = val_label.astype(np.float16)
#     # test_pairs = test_pairs.astype(np.float16)
#     # test_label = test_label.astype(np.float16)

#     # history=model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
#     #       batch_size=128,epochs=50,
#     #       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label))
    

#     model.save_weights('model_weights.h5')

#     train_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])

#     #计算训练集中的判定分数的均值
#     interscore=[]
#     intrascore=[]
#     for i in range(len(train_label)):
#         if train_label[i]==1:
#             interscore.append(train_pred[i])
#         else:
#             intrascore.append(train_pred[i])
#     interscore=np.mean(interscore)
#     intrascore=np.mean(intrascore)
#     print("类间训练集分数：",interscore)
#     print("类外训练集分数：",intrascore)

#     test_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

#     tr_acc = compute_accuracy(train_label, train_pred)
#     te_acc = compute_accuracy(test_label, test_pred)
#     print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#     print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
#     return test_pred,test_label


# def siamese_ori_final(data,target,num_classes):

#     pairs, label = create_pairs_incre_1(data, target,num_classes)
#     input_shape = (len(data[0]),len(data[0][0]))
   
#     model=create_siamese_network(input_shape)
#     model.load_weights('model_weights.h5')
    
#     pred = model.predict([pairs[:, 0], pairs[:, 1]])
#     return pred,label


