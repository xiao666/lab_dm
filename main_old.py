# -*- coding: utf-8 -*-
import numpy as np
import TS_cf as cf
import number_signal as cf2
import study_box_length as cf3
#import keyboard as cf
#import knn as cf
import matplotlib.pyplot as plt



#load train data
f1=open("Coffee_TRAIN")
train=np.genfromtxt(f1,delimiter=',')
f2=open("Coffee_TEST")
test=np.genfromtxt(f2,delimiter=',')
print "load end"

#study for # ensembled times
'''
#main part
div_1=3
div_2=div_1

cf.ensemble(50,train,test,div_1,div_2)
'''

#study for # length box
'''
acc_list=[]
for p in range(20):
    print p+1
    div_1=p+1
    div_2=div_1
    acc_list.append(cf3.ensemble(50,train,test,div_1,div_2))
x=range(1,len(acc_list)+1)

plt.plot(x,acc_list)
plt.show()
'''

#study for # selected signals

#main part
div_1=5
div_2=div_1

cf.ensemble(10,train,test,div_1,div_2)
