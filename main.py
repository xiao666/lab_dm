
import TS_cf as CF
import numpy as np
import random
import time
'''
##load train data.
#for UCR dataset

f1=open("Gun_Point_TRAIN")
train=np.genfromtxt(f1,delimiter=',')

f2=open("Gun_Point_TEST")
test=np.genfromtxt(f2,delimiter=',')
print "load end"

train_y=train[:,0]
train_x=train[:,1:len(train[0])]
test_y=test[:,0]
test_x=test[:,1:len(train[0])]

div_1=5
div_2=div_1
#start = time.clock()
CF.ensemble(10,train_x,train_y,test_x,test_y,div_1,div_2)
#end = time.clock()
#t=end-start

#print t


'''
#for typing dataset
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


#set hyperparameters and run
div_1=10
div_2=div_1
CF.ensemble(50,train_x,train_y,test_x,test_y,div_1,div_2)



'''
##this is only for projection demo

train_x,train_y,select_signal=CF.select_signal(train_x,train_y)
dist=CF.dist(train_x,select_signal)
CF.projection(dist,train_y)

'''

