import torch 
import torch.nn as nn
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
def Readdata(filename):
    data=sio.loadmat(filename)
    
    samples=data['samples']
    labels=data['labels']
    
    samples/=np.max(np.abs(samples))
    samples-=np.mean(samples,axis=0)
    
    x_train,x_test,y_train,y_test = train_test_split(samples,labels,test_size=0.2,random_state=0)
    
    return x_train,x_test,y_train,y_test
    
    
def TrainingAndTest(x_train,x_test,y_train,y_test):
    Hidden=5
    learning_rate=0.01
    num_in,dim_in=x_train.shape
    _,dim_out=y_train.shape
    test_num,_=x_test.shape
    model=nn.Sequential(nn.Linear(dim_in,Hidden),
                        nn.Softmax(),
                        nn.Linear(Hidden,dim_out))
   ###nn.Relu() or nn.Sigmoid() or nn.Softmax()
    
    loss_function = nn.MSELoss(reduction='sum')
    #train model para
    episode=5000
    batch_size=8
    iterations=int(num_in/batch_size)#每个回合的迭代次数
    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]
    for i in range(episode):
        loss_epoch=0
        for j in range(iterations):
            x=torch.Tensor(x_train[j*batch_size:(j+1)*batch_size])
            y=torch.Tensor(y_train[j*batch_size:(j+1)*batch_size])
            y_pred=model(x)
            loss=loss_function(y_pred,y)
            loss_epoch+=loss.item()
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for para in model.parameters():
                    para-=learning_rate*para.grad
     
        train_loss.append(loss_epoch/(iterations*batch_size))
        y_pred=model(torch.Tensor(x_train))
        #loss=loss_function(y_pred,torch.Tensor(y_train))
        #train_loss.append(loss.item())
        y_pred=torch.argmax(y_pred,1).numpy()
        #print('第',i,'次训练精度为',1-np.mean(y_pred!=np.argmax(y_train,1)))
        train_acc.append(1-np.mean(y_pred!=np.argmax(y_train,1)))
        ####每训练十次，测试一次
        if i%10==0:
             
            y_pred=model(torch.Tensor(x_test))
            y=torch.Tensor(y_test)
            loss=loss_function(y_pred,y)
            test_loss.append(loss.item()/test_num)
            y_pred=torch.argmax(y_pred,1).numpy()  
            test_acc.append(1-np.mean(y_pred!=np.argmax(y_test,1)))
            print('第',int(i/10),'次测试精度为',1-np.mean(y_pred!=np.argmax(y_test,1)))
             
    return train_loss,train_acc,test_loss,test_acc
    
    
def plot(loss,acc):
    fig,ax1=plt.subplots()
    ax1.plot(loss,color='tab:red',label='loss')
    ax2 = ax1.twinx()
    ax2.plot(acc,color='tab:blue',label='accuary')
    fig.show()

if __name__ == "__main__":
    filename='iris'
    x_train,x_test,y_train,y_test=Readdata(filename)
    train_loss,train_acc,test_loss,test_acc=TrainingAndTest(x_train,x_test,y_train,y_test)
    print('上图为训练精度与损失函数值变化图，下图为测试精度与损失函数值变化图')
    plot(train_loss,train_acc)
    plot(test_loss,test_acc)
    
    
    
    