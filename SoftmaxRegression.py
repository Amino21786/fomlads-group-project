import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse
def onehotvector(X):
    m=X.shape[0]
    onehot=scipy.sparse.csr_matrix((np.ones(m), (X, np.array(range(m)))))
    onehot=np.array(onehot.todense()).T
    return onehot


def softmax_stable(x):
    x-=np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
def softmax_implement(x,w):
    a=np.dot(x,w)
    ans=softmax_stable(a)
    return ans

def lossandgrad(x,y,w,r):
    addt=np.ones([x.shape[0],1])
    x=np.hstack((x,addt))
    m=x.shape[0]
    prob= softmax_implement(x, w)
    y_edit= onehotvector(y)
    loss=-(1/m)*np.sum(y_edit*np.log(prob))+(r/2)*np.sum(w*w)
    grad=-(1/m)*np.dot(x.T,(y_edit-prob))+r*w
    return loss, grad
def loss(x,y,w,r):
    addt=np.ones([x.shape[0],1])
    x=np.hstack((x,addt))
    m=x.shape[0]
    prob= softmax_implement(x, w)
    y_edit= onehotvector(y)
    loss=-(1/m)*np.sum(y_edit*np.log(prob))+(r/2)*np.sum(np.abs(w))
    return loss
def gradient(x,y,w,r):
    addt=np.ones([x.shape[0],1])
    x=np.hstack((x,addt))
    m=x.shape[0]
    prob= softmax_implement(x, w)
    y_edit= onehotvector(y)
    grad=-(1/m)*np.dot(x.T,(y_edit-prob))+r*w
    return grad


def gradient_descent(x,y,w,a,iter,r):
    loss_vals=[]
    for i in range(0,iter):
        losses =loss(x,y,w,r)
        grad= gradient(x,y,w,r)
        w=w-(a*grad)
        loss_vals.append(losses)
    return loss_vals, w

def predictionfromprobability(x,w):
    addt=np.ones([x.shape[0],1])
    x=np.hstack((x,addt))
    probability= softmax_implement(x,w)
    prediction= np.argmax(probability, axis=1)
    return probability, prediction



from modelconstruct import train_test_data
from modelconstruct import standardscaler
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
import time


np.random.seed(42)
def SoftmaxRegression(dataset, n, learning_rate, iterations, regularisation):
    
    x_train, Y_train, x_test, Y_test= train_test_data (dataset, n)
    start=time.process_time()
    X_train = standardscaler(x_train)
    X_test = standardscaler(x_test)
    w=np.random.rand(X_train.shape[1]+1,len(np.unique(Y_train)))
    loss_vals, final_weight= gradient_descent(X_train,Y_train, w, learning_rate, iterations,regularisation)
    #plt.figure()
    #plt.plot(loss_vals)
    probabilities, prediction=predictionfromprobability(X_train, final_weight)
    acc_train= accuracy(prediction, Y_train)
    test_prob, test_predict=predictionfromprobability(X_test, final_weight)
    acc_test = accuracy(test_predict, Y_test)
    confusion_mat= confusion_matrix(Y_train, prediction)
    pr_mat=precision_and_recall(confusion_mat)
    microf1= micro_average_f1_score(confusion_mat)
    macrof1= macro_average_f1_score(pr_mat)
    run_time = time.process_time()-start #time taken to compute all of it
    plt.show()

    return acc_test, microf1, macrof1

print("---------------------------------------------------------------------")
print('SoftmaxRegression: The accuracy, micro_f1, macro_f1 and the run_time are the following')
print(SoftmaxRegression('MobilePricingUpdated.csv',0.8, 0.001, 3000, 1))







