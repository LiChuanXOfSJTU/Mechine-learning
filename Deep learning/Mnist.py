# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:46:14 2018

@author: wretched
"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torchtest import Net
import torch
import torch.nn as nn

def Readdata(filename):
    data=sio.loadmat(filename)
    samples = data['fea']
    labels = data['gnd']
    x_train=samples[:60000]
    y_train=labels[:60000]
    x_test=samples[60000:]
    y_test=labels[60000:]
    
    return x_train,y_train,x_test,y_test

def TrainAndTest(x_train,y_train,x_test,y_test):
    num_in,dim_in=x_train.shape
    _,dim_out=y_train.shape
    
    test,_=x_test.shape
    
    
    #(N,C,W,H)
    x_train_torch=x_train.reshape(-1,1,28,28)
    x_test_torch=x_test.reshape(-1,1,28,28)
    
    y_train_torch=y_train.reshape(-1)
    y_test_torch=y_test.reshape(-1)
    
    #convert to tensor
    x_train=torch.Tensor(x_train_torch)
    x_test=torch.Tensor(x_test_torch)
    y_train=torch.Tensor(y_train_torch)
    y_test=torch.Tensor(y_test_torch)
    
    episode=5
    batch_size=64
    train_iterations=int(num_in/batch_size)#每个回合的迭代次数
    test_iterations=int(test/batch_size)
    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]
    
    net=Net()
    loss_function=nn.CrossEntropyLoss(reduction='sum')
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    for i in range(episode):
        train_loss_epoch=0
        train_acc_epoch=0
        test_loss_epoch=0
        test_acc_epoch=0
        for j in range(train_iterations):
            x=x_train[j*batch_size:(j+1)*batch_size]
            y=y_train[j*batch_size:(j+1)*batch_size]
            y_pred=net(x)
            
            loss=loss_function(y_pred,y.type(torch.long))
            train_loss_epoch+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred=torch.argmax(y_pred,1).numpy()
            y=y.numpy()
            acc=1-np.mean(y_pred!=y)
            train_acc_epoch+=acc
        print(train_acc_epoch/train_iterations)
        print(train_loss_epoch/(train_iterations*batch_size))
        train_acc.append(train_acc_epoch/train_iterations)
        train_loss.append(train_loss_epoch/train_iterations*batch_size)
        
        #测试
        for n in range(test_iterations):
            x=x_test[n*batch_size:(n+1)*batch_size]
            y=y_test[n*batch_size:(n+1)*batch_size]
            y_pre=net(x)
            loss=loss_function(y_pre,y.type(torch.long))
            test_loss_epoch+=loss.item()
            y_pre=torch.argmax(y_pre,1).numpy()
            y=y.numpy()
            acc=1-np.mean(y_pre!=y)
            test_acc_epoch+=acc
        print(test_acc_epoch/test_iterations)
        print(test_loss_epoch/(test_iterations*batch_size))
        test_loss.append(test_loss_epoch/(test_iterations*batch_size))
        test_acc.append(test_acc_epoch/test_iterations)
    return train_loss,train_acc,test_loss,test_acc
    
    
def plot(loss,acc):
    fig,ax1=plt.subplots()
    ax1.plot(loss,color='tab:red',label='loss')
    ax2 = ax1.twinx()
    ax2.plot(acc,color='tab:blue',label='accuary')
    fig.show()   
          

            


if __name__ == "__main__":
    filename='Mnist'
    x_train,y_train,x_test,y_test=Readdata(filename)
    train_loss,train_acc,test_loss,test_acc=TrainAndTest(x_train,y_train,x_test,y_test)
    plot(train_loss,train_acc)
    plot(test_loss,test_acc)
    
    
    