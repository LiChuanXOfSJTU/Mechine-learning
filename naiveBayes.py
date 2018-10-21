# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:39:09 2018

@author: 李传霄
"""
import numpy as np
import pandas as pd
import math

def read_data(filename):
    df=pd.read_excel(filename)
    return df

def MLE(GoodorBad):
    #极大似然估计
    List=[]
    result=[]
    List.append(np.mean(GoodorBad,axis=0))
    List.append(np.std(GoodorBad,axis=0))
    result.append(List[0][6])#密度的均值
    result.append(List[0][7])#含糖量的均值
    result.append(List[0][6])#密度的标准差
    result.append(List[0][7])#含糖量的标准差
    return result
    
    
    
    
        
def calProbability(train,test):
    Good=[]
    Bad=[]
    for k in range(len(train)):
        if train[k,8]==1:
            Good.append(train[k,:])
        else:
            Bad.append(train[k,:])
    Good=np.array(Good)
    Bad=np.array(Bad)
    deltaGood=MLE(Good)
    deltaBad=MLE(Bad)
    GoodNum=len(Good)
    BadNum=len(Bad)
    result=[]
    for i in range(3):
        GoodPro=(GoodNum+1)/(GoodNum+BadNum+2)#Laplace 修正
        BadPro=1-GoodPro
        for j in range(6):
            GoodPro=GoodPro*P(Good,test[i,j])
            BadPro=BadPro*P(Bad,test[i,j])
        GoodPro=GoodPro*1/(math.sqrt(2*math.pi)*deltaGood[1])*math.exp(-((test[i,6]-deltaGood[0])**2)/(2*(deltaGood[1]**2)))
        BadPro=BadPro*1/(math.sqrt(2*math.pi)*deltaBad[1])*math.exp(-((test[i,6]-deltaBad[0])**2)/(2*(deltaBad[1]**2)))
        result.append(GoodPro)
        result.append(BadPro)
    
    return result#6个元素，两个一组分别是1或0的概率


def acc(result,test):
    true=0
    false=0
    predict=[]
    k=0
    while k<5:
        if result[k]>result[k+1]:
            predict.append(1)
            k=k+2
        else:
            predict.append(0)
            k=k+2
    print("预测结果")
    print(predict)
    print("测试集")
    print(test)
    for i in range(3):
        if predict[i]==test[i,8]:
            true+=1
        else:
            false+=1
    return true/(true+false)
    
    
    


def P(GoodorBad,number):
    Num=len(GoodorBad)
    total=0
    for i in range(Num):
        for j in range(6):
            if GoodorBad[i,j]==number:
                total+=1
    return (total+1)/(Num+3) #laplace 修正
    
    
    


if __name__ == '__main__':
    filename='西瓜数据3.0.xls'
    data=read_data(filename)
    print(data)
    k=0
    totalACC=0
    while k<15:
        test=(data.iloc[k:k+3,:]).values
        row1=(data.iloc[0:k,:]).values
        row2=(data.iloc[k+3:,:]).values
        train=np.row_stack((row1,row2))
        result=calProbability(train,test)
        Acc=acc(result,test)
        print("本次精度为:")
        print(Acc)
        totalACC+=Acc
        k=k+3
        
        
    print("5次平均精度为")
    print(totalACC/5)
    
