import numpy as np
import TS_cf as cf
from random import randrange
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#load train data
f1=open("Gun_Point_TRAIN")
train=np.genfromtxt(f1,delimiter=',')

f2=open("Gun_Point_TEST")
test=np.genfromtxt(f2,delimiter=',')

print "load end"

#select 2 signals
select_index1=randrange(0,len(train))
t1=train[select_index1]
train=np.delete(train,(select_index1),axis=0)
select_index2=randrange(0,len(train))
t2=train[select_index2]
train=np.delete(train,(select_index2),axis=0)
select_signal=np.delete([t1,t2],(0),axis=1)


train_y=train[:,0]
train_x=np.delete(train,(0),axis=1)
train_dist=cf.dist(train_x,select_signal,train_y)

#projection
#-------------------change with different class label value
val_1=1
val_2=2
p1_idx=[index for index, val in enumerate(train_y) if val==val_1]
p2_idx=[index for index, val in enumerate(train_y) if val==val_2]
p1_dist=np.asmatrix(train_dist)[p1_idx]
p2_dist=np.asmatrix(train_dist)[p2_idx]
plt.plot(p1_dist[:,0],p1_dist[:,1],'bs',p2_dist[:,0],p2_dist[:,1],'g^')
plt.legend(["class 1","class 2"],numpoints=1)
plt.ylabel("1000* (dtw distance to signal %d)" %select_index2)
plt.xlabel("1000* (dtw distance to signal %d)" %select_index1)
plt.title("Gun_Point_train_Projection")
plt.show()

#projection for test data to the 2 selected signals
test_y=test[:,0]
test_x=np.delete(test,(0),axis=1)
test_dist=cf.dist(test_x,select_signal,test_y)
p1_idx=[index for index, val in enumerate(test_y) if val==val_1]
p2_idx=[index for index, val in enumerate(test_y) if val==val_2]
p1_dist=np.asmatrix(test_dist)[p1_idx]
p2_dist=np.asmatrix(test_dist)[p2_idx]
plt.plot(p1_dist[:,0],p1_dist[:,1],'bs',p2_dist[:,0],p2_dist[:,1],'g^')
plt.legend(["class 1","class 2"],numpoints=1)
plt.ylabel("1000* (dtw distance to signal %d)" %select_index2)
plt.xlabel("1000* (dtw distance to signal %d)" %select_index1)
plt.title("Gun_Point_test_Projection")
plt.show()