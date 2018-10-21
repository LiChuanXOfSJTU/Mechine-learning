# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:06:28 2018

@author: 李传霄
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

def read_data(filename):
    df=pd.read_excel(filename)
    return df

def SVMOFlinear(x_train,x_test,y_train,y_test):
    print("线性核函数")
    clf = svm.SVC(C=100, kernel='linear', decision_function_shape=None)
    clf.fit(x_train, y_train.ravel())
    print("训练精度")
    print (clf.score(x_train, y_train)) # 精度
    print("训练结果")
    y_predict = clf.predict(x_train)
    print(y_predict)
    print("测试精度")
    print (clf.score(x_test, y_test))
    print("测试结果")
    y_predict = clf.predict(x_test)
    print(y_predict)
    
def SVMOFRBF(x_train,x_test,y_train,y_test):
    print("高斯核函数")
    clf = svm.SVC(C=1000, kernel='rbf',gamma=0.001,decision_function_shape=None)
    
    clf.fit(x_train, y_train.ravel())
    print("训练精度")
    print (clf.score(x_train, y_train)) # 精度
    print("训练结果")
    y_predict = clf.predict(x_train)
    print(y_predict)
    print("测试精度")
    print (clf.score(x_test, y_test))
    print("测试结果")
    y_predict = clf.predict(x_test)
    print(y_predict)

def SVMOFSigmoid(x_train,x_test,y_train,y_test):
    print("Sigmoid核函数")
    clf = svm.SVC(C=1.0, kernel='sigmoid',gamma='auto',decision_function_shape=None)
    clf.fit(x_train, y_train.ravel())
    print("训练精度")
    print (clf.score(x_train, y_train)) # 精度
    print("训练结果")
    y_predict = clf.predict(x_train)
    print(y_predict)
    print("测试精度")
    print (clf.score(x_test, y_test))
    print("测试结果")
    y_predict = clf.predict(x_test)
    print(y_predict)

def SVMOFPoly(x_train,x_test,y_train,y_test):
    print("多项式核函数")
    clf = svm.SVC(C=1.0, kernel='poly',degree=3,gamma='auto',decision_function_shape=None)
    clf.fit(x_train, y_train.ravel())
    print("训练精度")
    print (clf.score(x_train, y_train)) # 精度
    print("训练结果")
    y_predict = clf.predict(x_train)
    print(y_predict)
    print("测试精度")
    print (clf.score(x_test, y_test))
    print("测试结果")
    y_predict = clf.predict(x_test)
    print(y_predict)
    

def GridSearch(x_train,y_train):
    tuned_parameters = [
                    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel':['sigmoid'],'C':[1,10,100,1000]},
                    {'kernel':['poly'],'C':[1,10,100,1000]}
                    ]
    score='precision'
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,scoring=score)
    clf.fit(x_train,y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
    

if __name__ == '__main__':
    filename='西瓜数据3.0.xls'
    data=read_data(filename)
    
    x,y=data.iloc[:,0:8],data.iloc[:,8]
    x=x.values
    y=y.values
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
   
    print("训练集数据")
    print(x_train)
    print(y_train)
    print("测试集数据")
    print(x_test)
    print(y_test)
    print()
    GridSearch(x_train,y_train)#grid 调参
    print()
    SVMOFlinear(x_train,x_test,y_train,y_test)#线性核函数
    print()
    SVMOFRBF(x_train,x_test,y_train,y_test)#高斯核函数
    print()
    SVMOFSigmoid(x_train,x_test,y_train,y_test)#sigmoid 核函数
    print()
    SVMOFPoly(x_train,x_test,y_train,y_test)
    
    
    
    
    
    
    


