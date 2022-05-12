# -*- coding:utf-8 -*-

import numpy as np
import nltk
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics

#测试样本数
N=100


def load_user_cmd(filename):
    cmd_list=[]
    dist_max=[]
    dist_min=[]
    dist=[]
    with open(filename) as f:
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            dist.append(line)
            i+=1
            if i == 100:
                cmd_list.append(x)
                x=[]
                i=0

    # print(FreqDist(dist))
    fdist = list(FreqDist(dist).keys())
    # print(fdist)
    dist_max=set(fdist[0:50])
    dist_min = set(fdist[-50:])
    # print(dist_max)
    # print(dist_min)
    return cmd_list,dist_max,dist_min

def get_user_cmd_feature(user_cmd_list,dist_max,dist_min):
    user_cmd_feature=[]
    for cmd_block in user_cmd_list:
        # print(cmd_block)
        # print(set(cmd_block))
        f1=len(set(cmd_block))
        fdist = list(FreqDist(cmd_block).keys())
        f2=fdist[0:10]
        f3=fdist[-10:]
        f2=len(set(f2) & set(dist_max))
        f3=len(set(f3)&set(dist_min))
        x=[f1,f2,f3]
        # print(x)
        user_cmd_feature.append(x)
    return user_cmd_feature

def get_label(filename,index=0):
    x=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            # print(int(line.split()[2]))
            # print(index)
            x.append(int(line.split()[index]))
    return x

if __name__ == '__main__':
    user_cmd_list,user_cmd_dist_max,user_cmd_dist_min=load_user_cmd("data/MasqueradeDat/User2")
    user_cmd_feature=get_user_cmd_feature(user_cmd_list,user_cmd_dist_max,user_cmd_dist_min)
    labels=get_label("data/MasqueradeDat/label.txt",1)
    # print(labels[50:])
    # print(labels)
    y=[0]*50+labels

    x_train=user_cmd_feature[0:N]
    y_train=y[0:N]

    x_test=user_cmd_feature[N:150]
    y_test=y[N:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict=neigh.predict(x_test)

    score=np.mean(y_test==y_predict)*100

    #print y
    #print y_train
    print(y_test)
    print(y_predict)
    print(score)

    print(classification_report(y_test, y_predict))

    print(metrics.confusion_matrix(y_test, y_predict))
