#!/usr/bin/env python
# coding: utf-8

# In[8]:


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from math import log


# In[9]:


def binarization(dataSet):
    'data preprocess(binarization) '
    dataSet = np.where(dataSet > 0, 1, 0)
    return dataSet   


# In[10]:


def getParameters(trainDataSet, trainLabel, alpha):

    numTrainData, numFeature = np.shape(trainDataSet)    #no. of train data and no. of features
    numClass1 = sum(trainLabel == 1)                     #no. of class 1, good email
    numClass0 = sum(trainLabel == 0)                     #no. of class 0, spam email
    pClass1 = numClass1/numTrainData                     #Lambda_ML
    pClass0 = 1 - pClass1                                #1-λ_ML

    Nj1 = np.zeros((numFeature), dtype = 'int')    #store N_j,1, array size:(1,57)
    Nj0 = np.zeros((numFeature), dtype = 'int')    #store N_j,0, array size:(1,57), int so that later can be used to calculate
    

    a = trainDataSet* trainLabel                           #keep features for good emails
    b = trainDataSet* (np.where(trainLabel == 0, 1, 0))    #keep features for spam emails
    #print(a)
    #print(b)

    for i in range(numFeature):
        Nj1[i] = sum(a[:, i] == 1)       #add up all x_j,1 (x_j,1 = 1 for good mail in a) and get N_j,1
        Nj0[i] = sum(b[:, i] == 1)       #add up all x_j,0 (x_j,0 = 1 for spam mail in b) and get N_j,0

    p1Class1 = (Nj1 + alpha)/(numClass1 + 2*alpha)
    p1Class0 = (Nj0 + alpha)/(numClass0 + 2*alpha)

    return p1Class1, p1Class0, pClass1, pClass0             #all parameters needed are obtained at this step


# In[11]:


def BetaNBclassifier(testDataSet, p1Class1, p1Class0, pClass1, pClass0):   #classifier to get predicted classes

    numTestData, numFeature = np.shape(testDataSet)
    predictedLabel = np.zeros((numTestData, 1), dtype = 'int')       #a column(N,1) to store predicted class 0/1

    for i in range(numTestData):          #for every row in dataSet
        p1 = log(pClass1)
        p0 = log(pClass0)
        x = testDataSet[i]* p1Class1 + np.where(testDataSet[i] == 0, 1, 0)* (1 - p1Class1)
        y = testDataSet[i]* p1Class0 + np.where(testDataSet[i] == 0, 1, 0)* (1 - p1Class0)
        
        for j in range(numFeature):      #finally, get p(y=1) and p(y=0)
            p1 += log(x[j])
            p0 += log(y[j])
        if p1 > p0:                     # classify
            predictedLabel[i] = 1    
        else:
            predictedLabel[i] = 0         

    return predictedLabel               #get predicted classes (N,1)



# In[12]:



if __name__ == "__main__":
    
    spamData = 'spamData.mat'
    data = scio.loadmat(spamData)            #Load spam data
    
    Xtrain = binarization(data['Xtrain'])
    Xtest = binarization(data['Xtest'])       #invoke function to binarize dataset
    ytrain = data['ytrain']
    ytest = data['ytest']
    
    errorRateList_test = []       # a list for each errorRate for corresponding alpha
    errorRateList_train = []

    for alpha in np.arange(0, 100.5, 0.5):
        p1Class1, p1Class0, pClass1, pClass0 = getParameters(Xtrain, ytrain, alpha)#invoke function to get parameters needed
        
        ypredict_test = BetaNBclassifier(Xtest, p1Class1, p1Class0, pClass1, pClass0)       #invoke function to classify
        ypredict_train = BetaNBclassifier(Xtrain, p1Class1, p1Class0, pClass1, pClass0) 
        
        accuracy_test = sum(ypredict_test == ytest)/len(ytest)   #correctly predicted/total number
        accuracy_train = sum(ypredict_train == ytrain)/len(ytrain)
        
        errorRate_test = 1 - accuracy_test  
        errorRate_train = 1 - accuracy_train               #error rate = 1- accuracy
       
        errorRateList_test.append(errorRate_test)                #every loop with a different alpha, put errorRate in the list
        errorRateList_train.append(errorRate_train)
        #print the error for each alpha
        print(" alpha : ", alpha, "test_error: ", errorRate_test, "train_error:",errorRate_train)
        



 


# In[13]:


#plot error to alpha
alpha = np.arange(0, 100.5, 0.5)

plt.figure(figsize=(6,4),dpi=100) 
plt.plot(alpha,errorRateList_test,color="#F08080",label=u'test error')   #plot test_error to alpha                    
plt.plot(alpha,errorRateList_train,color="seagreen",label=u'train error') #plot train_error to alpha

plt.xlabel('alpha', fontsize=20)              
plt.ylabel('Error Rate', fontsize=20)
plt.title('test and training error versus α(Beta-binomial)')
plt.grid()
plt.legend(loc="upper left")
plt.show()

