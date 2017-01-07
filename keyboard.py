# -*- coding: utf-8 -*-
#from dtw import dtw
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scipy import stats
import TS_cf as cf

###keyboard data load in====================================================================================================================
data=[]
temp_line=[]
f=open("keyboard.txt")
for line in f.readlines():
    x=line.split()
    for i in range(len(x)):
        x[i]=int(x[i])
    data.append(x)
print type(data)
#there are 200 datas
train=data[50:200]
test=data[30:40]
train_x,train_y=[],[]

test_x,test_y=[],[]
for i in range(len(test)):
    test_y.append(test[i][0])
test_x=test

for i in range(len(train)):
    train_y.append(train[i][0])
    del train[i][0]
train_x=train



f.close()


print "load end"
#load end ====================================================================================================================

sig_1=train_x[5]
sig_2=train_x[6]
selected_signal=[]
selected_signal.append(sig_1)
selected_signal.append(sig_2)
print len(selected_signal)
print "siganl end"

train_x=np.delete(train_x,(5,6))
train_y=np.delete(train_y,(5,6))

train_x=train_x[0:-1]
train_y=train_y[0:-1]

d1=[]
d2=[]
for i in range(len(train_x)):
    d1.append((fastdtw(train_x[i],sig_1)*1000)[0])
    d2.append((fastdtw(train_x[i],sig_2)*1000)[0])
train_dist=np.transpose([d1,d2,train_y])

d1=[]
d2=[]
for i in range(len(test_x)):
    d1.append((fastdtw(test_x[i],sig_1)*1000)[0])
    d2.append((fastdtw(test_x[i],sig_2)*1000)[0])
test_dist=np.transpose([d1,d2])



#distance - projection end ====================================================================================
div_1,div_2=15,15
box_number,x,y=cf.box(train_dist,div_1,div_2)
train_dist=np.append(train_dist,box_number,axis=1)
train_dist=np.delete(train_dist,(0,1),axis=1)#array label, box_number

prob=cf.bayes(train_dist,div_1,div_2)
print prob
print "box labeled"
# box are all labelled====================================================================================

test_box=[]
for i in range(len(test)):
    test_box.append(cf.judge_box(test_dist[i][0],test_dist[i][1],x,y,div_1,div_2))
print test_box
label=[]
for h in range(len(test_box)):
    label.append(prob[test_box[h]])
print label
final=[]
acc=0.0
n=0.0
print test_y
for i in range(len(label)):
    if (int(label[i])==int(test_y[i])):
        n=n+1
    else:
        pass
acc=n/len(label)
print acc