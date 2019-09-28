#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as scio
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import log


# In[2]:


def logTransform(dataSet):
    'preprocessing data by log transform'
    dataSet = np.log(dataSet + 0.1)
    return dataSet


# In[4]:


#concatenate x_i with 1 at the start.
def plusOne_data(dataSet):  
    numTrainData, numFeature = np.shape(dataSet)
    one=np.ones(numTrainData)
    data1 = np.c_[one,dataSet]
    #print(data1)
    return data1


# In[20]:


#train w,miu
def trainW(trainDataSet, trainLabel, lamb):
    numTrainData, numFeature = np.shape(trainDataSet)    #no. of train data and no. of features

    W = np.zeros((1,numFeature))      
    WT = np.transpose(W)        #WT_(D+1)*1,i.e,(58,1)
    
    X = trainDataSet            #X size :(N,D+1)
    XT = np.transpose(X)        #XT size:(D+1,N)
    
    S = np.identity((numTrainData))        #S :diag(N,N)
    I = np.zeros((numFeature,numFeature))  #I : Diag (58,58)with I[O,O]=0    
    for j in range(1,numFeature):
        I[j,j] = 1
    
    d = np.ones([numFeature , 1])          #D:（58，1）

    while np.max(np.abs(d)) > 0.0001:      # Convergence loop, in case of 'singular matrix'
        x = X.dot(WT)                      #x : (N,1)
        # print(x)
        miu = 1 / (1 + np.exp(-x))         #miu : (N,1)
        #print(miu)

        for j in range(numTrainData):     #set the diagnose value for S
            S[j, j] = miu[j, 0] * (1 - miu[j, 0])
        # print(S)

        g = (XT).dot((miu - trainLabel))  #g : (58,1)
        WT0 = WT.copy()
        WT0[0, 0] = 0                     #regularization should not apply to bias term
        g_reg = g + lamb * WT0
        # print('g',g)
        # print('greg',g_reg)

        H = np.dot((XT), S).dot(X)        #H : (58,58)
        H_reg = H + lamb * (I)
        H_inv = np.linalg.pinv(H_reg)
        # print(H)
        # print(H_reg)
        # print('hinv:',H_inv)

        d = -(H_inv).dot(g_reg)         #d: (58,1)
        # print('d: ',d, 'end')
        WT = WT + d                     # WT:(58,1)
        # print(WT,'END')
    #print('miu: ',miu)
    #print(np.shape(miu))
    return miu, WT


# In[21]:


# classifier
def LRclassifier(testDataSet,WT):
    numTestData, numFeature = np.shape(testDataSet)
    predictedLabel = np.zeros((numTestData, 1), dtype = 'int')   #array to store predicted labels Size:(N,1)
    x = testDataSet.dot(WT)
    miu = miu = 1 / (1 + np.exp(-x))
    for i in range(len(miu)):                     #classify label with calculated miu
        p1 = miu[i,0]
        p0 = 1 - p1

        if p1 > p0:
            predictedLabel[i,0] = 1    
        else:
            predictedLabel[i,0] = 0         

    return predictedLabel
    


# In[22]:



if __name__ == "__main__":
    
    spamData = 'spamData.mat'
    data = scio.loadmat(spamData)         #load spam data
    #print(data['Xtrain'])
    Xtrain = logTransform(data['Xtrain']) #log transform the dataset
    Xtrain = plusOne_data(Xtrain)         #then bias term added
    
    Xtest = logTransform(data['Xtest'])   #do the same process to test data
    Xtest = plusOne_data(Xtest)      
    
    ytrain = data['ytrain']
    ytest = data['ytest']
    
    errorRateList_test = [] 
    errorRateList_train = []
    
    lamb1= np.arange(1,11,1)
    lamb2 = np.arange(15,105,5)
    lamb = np.concatenate([lamb1,lamb2]) # range for lambda
    
    
    for L in lamb:
        miu, WT = trainW(Xtrain, ytrain, L)           #invoke function to obtain miu, WT
        
        ypredict_test = LRclassifier(Xtest, WT)       #invoke function to get predicted labels
        ypredict_train = LRclassifier(Xtrain,WT)
        
        count1 = 0
        for i in range(len(ytest)):
            if ypredict_test[i] == ytest[i]:
                count1+=1
            else:
                continue
        accuracy_test = count1/len(ytest)   #correctly predicted/total number
        
        count2 = 0
        for i in range(len(ytrain)):
            if ypredict_train[i] == ytrain[i]:
                count2+=1
            else:
                continue
        accuracy_train = count2/len(ytrain)
        
        
        errorRate_test = 1 - accuracy_test  
        errorRate_train = 1 - accuracy_train               #error rate = 1 - accuracy
       
        errorRateList_test.append(errorRate_test)                #for every loop of lambda, put in an arror rate
        errorRateList_train.append(errorRate_train)
        print(" lambda: ", L, " TEST_error: ", errorRate_test, "TRAIN_error:",errorRate_train)
        


# In[23]:


lamb1= np.arange(1,11,1)
lamb2 = np.arange(15,105,5)
lamb = np.concatenate([lamb1,lamb2])

plt.figure(figsize=(6,4),dpi=100) 
plt.plot(lamb,errorRateList_test,color="#F08080",label=u'test error')                       #both error rate plot
plt.plot(lamb,errorRateList_train,color="seagreen",label=u'train error') 

plt.xlabel('lambda', fontsize=20)              
plt.ylabel('Error Rate', fontsize=20)
#plt.xlim((0,100)) 
plt.title('test and training error versus λ(Logistic Regression)')
plt.grid()
plt.legend(loc="lower center")
plt.show()

