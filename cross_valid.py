'''
import TS_cf as CF
import numpy as np
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt





###########################################################################################

##load train data.
#for UCR dataset

f1=open("ECG200_TRAIN")
train=np.genfromtxt(f1,delimiter=',')
print "load end"

or_train_y=train[:,0]
or_train_x=train[:,1:len(train[0])]
'''


'''
#10-fold cross-validation for selecting box length
model_acc=[]
for i in range(20):#20
    print "box length is divided into:",i+1
    div_1=i+1
    div_2=div_1
    k_fold = KFold(n_splits=5)#10
    #k_fold.get_n_splits(train)
    fold_acc=[]
    for train_index, test_index in k_fold.split(or_train_x):
        train_x, test_x = or_train_x[train_index], or_train_x[test_index]
        train_y, test_y = or_train_y[train_index], or_train_y[test_index]
        acc=CF.ensemble(10,train_x,train_y,test_x,test_y,div_1,div_2)
        fold_acc.append(acc)
    fold_avg_acc=np.average(fold_acc)
    print "current model avg acc:",fold_avg_acc
    model_acc.append(fold_avg_acc)
print "best model:",model_acc.index(np.max(model_acc))
print "best model acc:",np.max(model_acc)
x=range(1,len(model_acc)+1)
plt.plot(x,model_acc)
plt.show()
'''


