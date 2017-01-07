# -*- coding: utf-8 -*-
#from dtw import dtw
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

#use temp variable if a function term is calculated >1 times-----------------------


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
        #return (dtw(x,y,dist=lambda x,y: abs(x-y))[0])*1000
        return (fastdtw(x,y)*1000)[0]
    d1=[]
    d2=[]
    for i in range(len(train_x)):
        d1.append(d(train_x[i],select_signal[0]))
    for j in range(len(train_x)):
        d2.append(d(train_x[j],select_signal[1]))
    dist=np.transpose([d1,d2,train_y])
    return dist

def projection(dist,train_y):#ONLY IN 2 CLASS LABELS, BUT not in consider now----------------------
    val_1=1
    val_2=-1
    p1_idx=[index for index, val in enumerate(train_y) if val==val_1]
    p2_idx=[index for index, val in enumerate(train_y) if val==val_2]
    p1_dist=np.asmatrix(dist)[p1_idx]
    p2_dist=np.asmatrix(dist)[p2_idx]
    plt.plot(p1_dist[:,0],p1_dist[:,1],'bs',p2_dist[:,0],p2_dist[:,1],'g^')
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
def judge_box(x1,x2,x,y,div_1,div_2):# need a default for data out of borders?
    bx1=div_1+5-1
    bx2=div_2+5-1
    for i in range(div_1+5):
        #bx1=range(div_1+5)
        if x1>=i*x and x1<(i+1)*x:
            bx1=i
        else:
            pass
    for j in range(div_2+5):
        #bx2=range(div_2+5)
        if x2>=j*y and x2<(j+1)*y:
            bx2=j
        else:
            pass
    return bx2*(div_1+5)+bx1
    
def bayes(dist,div_1,div_2):#array label, box_number
    #p(c|Box)~~~p(Box|c)*p(c)=#c in B / # all
    #each Box, take max{p(c|BOX)} wrt c
    prob=[]
    temp=np.delete(dist,(0),axis=1)#box_number
    temp2=np.asarray(temp,dtype=int)
    for i in range((div_2+5)*(div_1+5)):
        temp_idx= [index for index, val in enumerate(temp2.tolist()) if val==[i]]
        if(len(temp_idx)>0):
            temp_list=np.asarray(np.delete(dist[temp_idx],(1),axis=1),dtype=int).tolist()
            tttt=(stats.mode(temp_list)[0][0]).tolist()#return [2]
            prob=prob+tttt
        else:
            #print "ggggggggggggggggggggggggggggggggggggggggggggg"
            prob.append(99999)
    return prob#[,,,,,]


def test(test,selected_signal,prob,x,y,div_1,div_2):
    test_y=test[:,0]
    test_x=test[:,1:len(test[0])]
    #cal dtw dist
    dist_test=np.transpose(dist(test_x,selected_signal,test_y))

    #assign to boxes
    test_box=[]
    for i in range(len(test)):
        test_box.append(judge_box(dist_test[0][i],dist_test[1][i],x,y,div_1,div_2))
    #print test_box
    #predict
    label=[]
    acc=0.0
    for h in range(len(test_box)):
        label.append(prob[test_box[h]])
    acc=cal_acc(label,test_y)
    print "round acc is :",acc
    
    return label,acc
    


def cal_acc(predict,true):
    acc=0.0
    for p in range(len(true)):
        if predict[p]==true[p]:
            acc=acc+1
    acc=acc/len(true)
    return acc


def train(train,test_data,div_1,div_2):
    train,selected_signal=select_signal(train)
    train_y=train[:,0]
    train_x=np.delete(train,(0),axis=1)
    #test_y=test[:,0]
    
    temp_d=dist(train_x,selected_signal,train_y)#array x1,x2,label
    
    #projection(temp_d,train_y)
    box_number,x,y=box(temp_d,div_1,div_2)
    temp_d=np.append(temp_d,box_number,axis=1)
    temp_d=np.delete(temp_d,(0,1),axis=1)#array label, box_number

    prob=bayes(temp_d,div_1,div_2)
    #print prob
    
    predict_label,acc=test(test_data,selected_signal,prob,x,y,div_1,div_2)
    '''


    #knn================================================================================================================================================
    test_y=test_data[:,0]
    test_x=np.delete(test_data,(0),axis=1)
    
    a_d=dist(train_x,selected_signal,train_y).tolist()
    b_d=dist(test_x,selected_signal,test_y).tolist()
    a_d=(np.delete(a_d,(-1),axis=1)).tolist()
    b_d=(np.delete(b_d,(-1),axis=1)).tolist()
    
    neigh=KNeighborsClassifier(n_neighbors=1)
    neigh.fit(a_d,train_y)
    predict_label222=neigh.predict(b_d)
    '''
    #=====================================================================================================================================================
    return predict_label

def ensemble(times,train_data,test,div_1,div_2):
    temp_label=[]
    label=[]
    test_y=test[:,0]
    #test_y=test[:][0]
    
    for i in range(times):
        print "round ",i+1
        temp_label.append(train(train_data,test,div_1,div_2))
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
