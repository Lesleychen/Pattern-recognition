#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from math import log


# In[2]:


def logTransform(dataSet):
    'preprocessing data by log transform'
    dataSet = np.log(dataSet + 0.1)
    return dataSet


# In[14]:


def prior_mean_var(trainDataSet, trainLabel):   #a function to calculate prior, mean and variance

    numTrainData, numFeature = np.shape(trainDataSet)         #N, 57features
    numClass1 = sum(trainLabel == 1)                          #N1
    pClass1 = numClass1/numTrainData                          #N1/N
    
    Xj1 = trainDataSet[np.where(trainLabel[:, 0] == 1)]       #rows in dataset in class1
    Xj0 = trainDataSet[np.where(trainLabel[:, 0] == 0)]       #rows in dataset in class0
    #print('Xj1: ',Xj1)
    #print(np.shape(Xj1))
    mean = np.row_stack((np.mean(Xj0, axis = 0), np.mean(Xj1, axis = 0)))
    #print('mean: ',mean)
    #print(np.shape(mean))
    #mean of each feature in class0 & mean of feature in class1  size:(2,57)
    var = np.row_stack((np.var(Xj0, axis = 0), np.var(Xj1, axis = 0)))
    #print('var: ',var)
    #print(np.shape(var))
    #var of feature in class0 & var of data in class1 size:(2,57)

    return pClass1, mean, var


# In[15]:


def GaussianNBClassifier(testDataSet, pclass1, Mean, Var):

    numTestData, numFeature = np.shape(testDataSet)            #N, 57
    predictLabel = np.zeros((numTestData, 1), dtype = 'int')   #Array(N,1) to store predicted labels 0/1

    for i in range(numTestData):  #for every row in test data
        p1 = log(pclass1)
        p0 = log(1 - pclass1)
        for j in range(numFeature):
            x = (np.exp(-(testDataSet[i][j] - Mean[1][j])**2/(2*Var[1][j])))/(np.sqrt(2*np.pi*Var[1][j]))
            y = (np.exp(-(testDataSet[i][j] - Mean[0][j])**2/(2*Var[0][j])))/(np.sqrt(2*np.pi*Var[0][j]))
            if x > 0:
                p1 += log(x)
            else:
                p1 += 0

            if y > 0 :
                p0 += log(y)
            else:
                p0 += 0 
    
        if p1 > p0:              #classify
            predictLabel[i] = 1
        else:
            predictLabel[i] = 0

    return predictLabel           #return the list of predicted labels(N,1)


# In[13]:


if __name__ == "__main__":
    
    spamData = 'spamData.mat'
    data = scio.loadmat(spamData)         #Load spam data
    Xtrain = logTransform(data['Xtrain'])
    Xtest = logTransform(data['Xtest'])
    ytrain = data['ytrain']
    ytest = data['ytest']
    
    pClass1, mean, var = prior_mean_var(Xtrain, ytrain)  #using train Data to get prior, mean, variance
    print(mean)
    print(var)
    
    ypredict_test = GaussianNBClassifier(Xtest, pClass1, mean, var)    #invoke function to get predicted labels list
    accuracy_test = sum(ypredict_test == ytest)/len(ytest)             #accuracy
    error_test = 1-accuracy_test                                       #error
    print('test error: ',error_test)
    
    ypredict_train = GaussianNBClassifier(Xtrain, pClass1, mean, var)
    accuracy_train = sum(ypredict_train == ytrain)/len(ytrain)
    error_train = 1-accuracy_train
    print('train error: ',error_train)
    
    

