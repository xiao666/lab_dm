# -*- coding: utf-8 -*-
import numpy as np
import TS_cf as cf
#import keyboard as cf
#import knn as cf

#load train data
f1=open("Gun_Point_TRAIN")
train=np.genfromtxt(f1,delimiter=',')
f2=open("Gun_Point_TEST")
test=np.genfromtxt(f2,delimiter=',')
print "load end"



'''

#keyboard data load in
data=[]
temp_line=[]
f=open("keyboard.txt")
for line in f.readlines():
    x=line.split()
    for i in range(len(x)):
        x[i]=int(x[i])
    data.append(x)

print len(data)
#there are 200 datas
train=data[0:139]
test=data[140:200]
'''
#main part
div_1=10
div_2=div_1

cf.ensemble(30,train,test,div_1,div_2)
#cf.ensemble(50,train,test)


