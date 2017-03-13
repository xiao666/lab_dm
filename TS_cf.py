# -*- coding: utf-8 -*-
#from dtw import dtw
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
import time

all_time=[]

def select_signal(train_x,train_y):
    #select 2 random signals
    select_idx=random.sample(range(len(train_x)),2)
    select_1=select_idx[0]
    select_2=select_idx[1]

    t1=train_x[select_1]
    t2=train_x[select_2]

    train_x=np.delete(train_x,(select_1,select_2),axis=0)
    train_y=np.delete(train_y,(select_1,select_2),axis=0)
    select_sig=[t1,t2]

    return train_x,train_y,select_sig

def dist(train_x,select_signal):#dtw is amplfied with 1,000
    def d(x, y):
        return (fastdtw(x,y)*1000)[0]
    d1=[]
    d2=[]
    for i in range(len(train_x)):
        d1.append(d(train_x[i],select_signal[0]))
        d2.append(d(train_x[i],select_signal[1]))
    dist=np.transpose([d1,d2])
    return dist

def projection(dist,train_y):#ONLY IN 2 CLASS LABELS, BUT not in consider now----------------------
    val_1=1
    val_2=2
    val_3=3
    val_4=4
    p1_idx=[index for index, val in enumerate(train_y) if val==val_1]
    p2_idx=[index for index, val in enumerate(train_y) if val==val_2]
    p3_idx=[index for index, val in enumerate(train_y) if val==val_3]
    p4_idx=[index for index, val in enumerate(train_y) if val==val_4]
    p1_dist=np.asmatrix(dist)[p1_idx]
    p2_dist=np.asmatrix(dist)[p2_idx]
    p3_dist=np.asmatrix(dist)[p3_idx]
    p4_dist=np.asmatrix(dist)[p4_idx]
    plt.plot(p1_dist[:,0],p1_dist[:,1],'bs',p2_dist[:,0],p2_dist[:,1],'g^',p3_dist[:,0],p3_dist[:,1],'r--',p4_dist[:,0],p4_dist[:,1],'ro')
    plt.show()
    return p1_dist,p2_dist

def box(dist,div_1,div_2):
    dist2=np.transpose(dist)
    x=max(dist2[0])/div_1 #length of a box 
    y=max(dist2[1])/div_2
    box_number=[]
    for idx in range(len(dist)):
        box_number.append(judge_box(dist[idx][0],dist[idx][1],x,y,div_1,div_2))

    return np.reshape(box_number,(-1,1)),x,y

# x,y is unit length
def judge_box(x1,x2,x,y,div_1,div_2):
    # need a default for data out of borders?
    #box number from 0~(d1+5)*(d2+5)
    bx1=div_1+5-1
    bx2=div_2+5-1
    for i in range(div_1+5):
        if x1>=i*x and x1<(i+1)*x:
            bx1=i
        else:
            pass
    for j in range(div_2+5):
        if x2>=j*y and x2<(j+1)*y:
            bx2=j
        else:
            pass
    return bx2*(div_1+5)+bx1
    
def bayes(train_box,y_label,div_1,div_2):#array label, box_number
    #p(c|Box)~~~p(Box|c)*p(c)=#c in B / # all
    #each Box, take max{p(c|BOX)} wrt c
    prob=[]
    for i in range((div_2+5)*(div_1+5)):
        temp_idx= [index for index, val in enumerate(train_box) if val==[i]]

        if(len(temp_idx)>0):
            temp_list=[y_label[m] for m in temp_idx]
            #temp_list=y_label[np.asarray(temp_idx)].tolist()
            tttt=(stats.mode(temp_list)[0][0]).tolist()#return [2]
            prob.append(tttt)
        else:
            prob.append(99999)
    return prob


def test(test_x,test_y,selected_signal,prob,x,y,div_1,div_2):
    start0 = time.clock()
    #cal dtw dist
    dist_test=np.transpose(dist(test_x,selected_signal))

    #assign to boxes
    test_box=[]
    for i in range(len(test_x)):
        test_box.append(judge_box(dist_test[0][i],dist_test[1][i],x,y,div_1,div_2))
    #print test_box
    #predict
    label=[]
    acc=0.0
    for h in range(len(test_box)):
        label.append(prob[test_box[h]])
    #acc=cal_acc(label,test_y)
    end0 = time.clock()
    #print "consumed time:"
    #print end0-start0
    t_temp=end0-start0
    all_time.append(t_temp)
    #print "round acc is :",acc

    return label,acc
    


def cal_acc(predict,true):
    acc=0.0
    for p in range(len(true)):
        if predict[p]==true[p]:
            acc=acc+1
    acc=acc/len(true)
    return acc


def train(train_x,train_y,test_x,test_y,div_1,div_2):
    #starttime=datetime.datetime.now()

    train_x,train_y,selected_signal=select_signal(train_x,train_y)

    temp_d=dist(train_x,selected_signal)
    
    #projection(temp_d,train_y)
    box_number,x,y=box(temp_d,div_1,div_2)

    prob=bayes(box_number,train_y,div_1,div_2)

    
    predict_label,acc=test(test_x,test_y,selected_signal,prob,x,y,div_1,div_2)
    #endtime = datetime.datetime.now()
    #print (endtime - starttime).seconds
    '''


    #knn==============
    
    a_d=dist(train_x,selected_signal,train_y).tolist()
    b_d=dist(test_x,selected_signal,test_y).tolist()
    a_d=(np.delete(a_d,(-1),axis=1)).tolist()
    b_d=(np.delete(b_d,(-1),axis=1)).tolist()
    
    neigh=KNeighborsClassifier(n_neighbors=1)
    neigh.fit(a_d,train_y)
    predict_label222=neigh.predict(b_d)
    '''
    #==================
    return predict_label

def ensemble(times,train_x,train_y,test_x,test_y,div_1,div_2):
    temp_label=[]
    label=[]

    for i in range(times):
        print "round ",i+1
        temp_label.append(train(train_x,train_y,test_x,test_y,div_1,div_2))
    temp_label=np.transpose(temp_label)
    accu_acc=[]
    for k in range(times):
        for i in range(len(test_x)):
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
    #print accu_acc
    print "final acc is : ",accu_acc[-1]
    x=range(1,len(accu_acc)+1)
    plt.plot(x,accu_acc)
    plt.show()
    print np.sum(all_time)
    return accu_acc
