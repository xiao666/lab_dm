'''
# -*- coding: utf-8 -*-
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
import time
import random


def cal_acc(predict,true):
    acc=0.0
    for p in range(len(true)):
        if predict[p]==true[p]:
            acc=acc+1
    acc=acc/len(true)
    return acc


#equal length time series data

f1=open("ECG200_TRAIN")
train=np.genfromtxt(f1,delimiter=',')

f2=open("ECG200_TEST")
test=np.genfromtxt(f2,delimiter=',')
print "load end"

train_y=train[:,0]
train_x=train[:,1:len(train[0])]
test_y=test[:,0]
test_x=test[:,1:len(train[0])]

'''
'''
#typing data (uneuqual length)
print "typing knn"
data=[]
temp_line=[]
f=open("keyboard.txt")
for line in f.readlines():
    x=line.split()
    for i in range(len(x)):
        x[i]=int(x[i])
    data.append(x)

#there are 200 datas
random.shuffle(data)
train=data[0:100]
test=data[100:200]

train_x,train_y=[],[]
test_x,test_y=[],[]


for i in range(len(test)):
    test_y.append(test[i][0])
    del test[i][0]
test_x=test

for i in range(len(train)):
    train_y.append(train[i][0])
    del train[i][0]#after append y label, delete the label
train_x=train
f.close()
'''

'''
start=time.clock()
w, h = len(train_x), len(test_x);
dist = [[0 for x in range(w)] for y in range(h)] 
label=[]
for i in range(len(test_x)):
    for j in range(len(train_x)):
        dist[i][j]=fastdtw(test_x[i],train_x[j])[0]
for k in range(len(dist)):
    idx=dist[k].index(np.min(dist[k]))
    label.append(train_y[idx])

acc=cal_acc(label,test_y)
print acc
end=time.clock()
print end-start
'''