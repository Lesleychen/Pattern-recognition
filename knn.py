#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io as scio
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import log


# In[3]:


def logTransform(dataSet):
    'preprocessing data by log transform'
    dataSet = np.log(dataSet + 0.1)
    return dataSet


# In[4]:


def distance(a,b):            #a function to obtain distance of two arrays
    c = a - b
    dist_i = (np.sum(c**2))**0.5
    return dist_i
    
    


# In[5]:


if __name__ == "__main__":
    
    spamData = 'spamData.mat'
    data = scio.loadmat(spamData)         #type: dict
    
    Xtrain = logTransform(data['Xtrain'])
    Xtest = logTransform(data['Xtest'])       #log transform the traning data and test data
    ytrain = data['ytrain']
    ytest = data['ytest']                  #type: numpy.ndarray     
    
    numTrainData,numFeature = np.shape(Xtrain)
    numTestData,numFeatureTest = np.shape(Xtest)
    
    dist = np.zeros((numTestData,numTrainData))    #dist is an array to store the distance values
    dist_train = np.zeros((numTrainData,numTrainData))
    
    errorRateList_test = [] 
    errorRateList_train = []
    
    ypredict_test = np.zeros((numTestData,1))
    ypredict_train = np.zeros((numTrainData,1))
    
    for i in range(numTestData):                   # a loop to calculate distance between each test data and train data
        for j in range(numTrainData):
            dist[i][j] = distance(Xtest[i],Xtrain[j])
            
    dist_sort = argsort(dist)                     #sort dist from small to large and get an array of index
    
    for i in range(numTrainData):                 # a loop to calculate distance between each train data and traindata
        for j in range(numTrainData):
            dist_train[i][j] = distance(Xtrain[i],Xtrain[j])
            
    dist_sort_train = argsort(dist_train)        #sort dist_train from small to large and get an array of index
    
    
    
    k1= np.arange(1,11,1)
    k2 = np.arange(15,105,5)
    K = np.concatenate([k1,k2])   #range for K
    
    for k in K: 
        #print('k: ',k)
        minimumk_order = np.zeros((1,k),dtype='int')           # array to store the indexes of the first kth least distance
        minimumk_order_train = np.zeros((1,k),dtype='int')
        for i in range(numTestData):                           #for each row in test data, predict p
            for s in range(k):
                minimumk_order[0][s] = dist_sort[i][s]
            #array of kth least distance
            kc = sum(ytrain[minimumk_order[0]] == 1)
            #print('kc: ',kc)
            p = kc/k
            if p<0.5:
                ypredict_test[i] = 0
            else:
                ypredict_test[i] = 1
            #print(ypredict_test)
            
            
        for i in range(numTrainData):                         #same process for train data
            #print(i)
            for s in range(k):
                minimumk_order_train[0][s] = dist_sort_train[i][s]
            #print(minimumk_order)
            kc = sum(ytrain[minimumk_order_train[0]] == 1)
            #print('kc: ',kc)
            p = kc/k
            if p<0.5:
                ypredict_train[i] = 0
            else:
                ypredict_train[i] = 1
            #print(ypredict_test)
        trainAccuracy = sum(ypredict_train == ytrain)/len(ytrain)   #correctly predicted/total
        testAccuracy = sum(ypredict_test == ytest)/len(ytest)
        errorRate_train = 1 - trainAccuracy                         #error rate
        errorRate_test = 1 - testAccuracy
        errorRateList_train.append(errorRate_train)                #for every loop of k, put in corresponding error
        errorRateList_test.append(errorRate_test)
        print(" K is: ", k, " error rate of test: ", errorRate_test, "error rate of train:",errorRate_train)
        


# In[6]:


#plot
plt.figure(figsize=(6,4),dpi=100) 
plt.plot(K,errorRateList_test,color="#F08080",label=u'test error')                       #both error rate plot
plt.plot(K,errorRateList_train,color="seagreen",label=u'train error') 

plt.xlabel('K', fontsize=20)              
plt.ylabel('Error Rate', fontsize=20)
plt.title('test and training error versus K(KNN)')
plt.grid()
plt.legend(loc="lower center")
plt.show()

