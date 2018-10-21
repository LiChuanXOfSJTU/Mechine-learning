# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:47:56 2018

@author: 李传霄
"""

import numpy as np
import pandas as pd
import math
def read_data(filename):
    df=pd.read_excel(filename)
    return df

def Sigmoid(x,y,w): 
    menber=len(x) #训练集个数
   
    n=len(x[0])   #属性个数
    
    result=0
    for i in range(menber):
        s=0
        for j in range(n):
            s+=x[i][j]*w[j]
           
        result+=-y[i]*(s)+math.log(1+math.exp(s))
   
    return result

def p1(x,w):
    #后验概率
    wx=0
    for i in range(len(x)):
        wx+=w[i]*x[i]
    return math.exp(wx)/(1+math.exp(wx))
    
    



def Derivative(x,y,w):
    D=np.zeros(len(x[0]))
    for i in range(len(x)):
        D+=x[i,:]*(y[i]-p1(x[i,:],w))
    return -D
        



def gradienet(x,y,w,error,n):
    i=0
    h=0.001
    while i<n:
        start1=Sigmoid(x,y,w)
        delta=Derivative(x,y,w)
        w=w-h*delta
        start2=Sigmoid(x,y,w)
        if abs(start1-start2)<error:
            print('梯度下降迭代',i,'步')
            print("w矩阵为")
            print(w)
            return w
            break
        i+=1
      
        

def test(x_test,w_trained):
    y=[]
    result=0
    for i in range(len(x_test)):
        s=0
        for j in range(len(x_test[i])):
            s+=x_test[i,j]*w[j]
        result=1/(1+math.exp(-s))
        y.append(result)
 
    y=np.array(y)
    
    return y
        
    
def acc(y_test,y_predict):
    true=0
    false=0
    for i in range(len(y_test)):
        if y_predict[i]>0.5:
            y_predict[i]=1
    
    for i in range(len(y_test)):
        if y_predict[i]==y_test[i,0]:
            true+=1
        else:
            false+=1
    acc=true/(true+false)
    print("测试集")
    print(y_test)
    print("预测结果")
    print(y_predict)
    return acc
    
    

    


if __name__ == '__main__':
    filename='西瓜数据3.0.xls'
    data=read_data(filename)
    print(data)
    x_train=(data.iloc[0:14,0:8]).values
    y_train=(data.iloc[0:14,8:]).values
    x_test=(data.iloc[14:,0:8]).values
    y_test=(data.iloc[14:,8:]).values
    
    
    column_train=np.ones(14)
    column_test=np.ones(3)
    x_train=np.column_stack((x_train,column_train))
    x_test=np.column_stack((x_test,column_test))
    
    w=np.ones(9) #9=8+1  
    error=math.exp(-8)#
    n=10000
    w_trained=gradienet(x_train,y_train,w,error,n)
    
    y_predict=test(x_test,w_trained)
    
    
    acc=acc(y_test,y_predict)
    print("精度为",acc)
    
    
   
    
    
   
   
    
    


