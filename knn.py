# -*- coding: utf-8 -*-
#from dtw import dtw
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier


def select_signal(train):
    select_index1=randrange(0,len(train))
    #select_index1=0
    t1=train[select_index1]
    train=np.delete(train,(select_index1),axis=0)
    select_index2=randrange(0,len(train))
    #select_index2=1
    t2=train[select_index2]
    train=np.delete(train,(select_index2),axis=0)
    select_signal=np.delete([t1,t2],(0),axis=1)
    #print select_index1, select_index2
    return train,select_signal

def dist(train_x,select_signal,train_y):#delete train_y---------------------------
    def d(x, y):
        return (fastdtw(x,y)*1000)[0]
    d1=[]
    d2=[]
    print train_x[0]
    for i in range(len(train_x)):
        d1.append(d(train_x[i],select_signal[0]))
    for j in range(len(train_x)):
        d2.append(d(train_x[j],select_signal[1]))
    dist=np.transpose([d1,d2,train_y])
    return dist



def cal_acc(predict,true):
    acc=0.0
    for p in range(len(true)):
        if predict[p]==true[p]:
            acc=acc+1
    acc=acc/len(true)
    return acc


def train(train,test_data):
    train,selected_signal=select_signal(train)
    train_y=train[:,0]
    train_x=train[:,1:-1]
    test_x=test_data[:,0]
    test_y=np.delete(test_data,(0),axis=1)
    
    train_d=dist(train_x,selected_signal,train_y)#array x1,x2,label
    test_d=dist(test_x,selected_signal,test_y)
    KNeighborsClassifier.fit(train_d,train_y)
    predict_label=KNeighborsClassifier.predict(test_d)
    return predict_label




def ensemble(times,train_data,test):
    temp_label=[]
    label=[]
    test_y=test[:,0]

    for i in range(times):
        print "round ",i+1
        temp_label.append(train(train_data,test))
    temp_label=np.transpose(temp_label)
    accu_acc=[]
    for k in range(times):
        for i in range(len(test)):
            no_99=temp_label[i][0:k+1]
            no_temp=[]
            for no in range(len(no_99)):
                if no_99[no]==99999:
                    no_temp.append(no)
                else:
                    pass
            no_99=np.delete(no_99,no_temp)
            tttt=(stats.mode(no_99)[0])
            label.append(tttt)
    #print label
        acc=cal_acc(label,test_y)
        label=[]
        accu_acc.append(acc)
    print accu_acc
    print "final acc is : ",accu_acc[-1]
    plt.plot(accu_acc)
    plt.show()


