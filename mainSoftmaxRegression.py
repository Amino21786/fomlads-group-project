import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse
def onehotvector(X):
    m=X.shape[0]
    onehot=scipy.sparse.csr_matrix((np.ones(m),(X,np.array(range(m)))))
    onehot=np.array(onehot.todense()).T
    return onehot

def softmax_alt(x):
    return np.exp(x) / np.sum(np.exp(x))
def softmax_stable(x):
    x-=np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
#could also use inbuilt from scipy
from scipy.special import softmax
def softmax_implement(x,w):
    a=np.dot(x,w)
    ans=softmax_stable(a)
    return ans

def lossandgrad(x,y,w):
    m=x.shape[0]
    prob= softmax_implement(x, w)
    y_edit= onehotvector(y)

    loss=(-1/m)*np.sum(y_edit*np.log(prob))+(1/2)*np.sum(w*w)
    grad=(-1/m)*np.dot(x.T,(y_edit-prob))+ w
    return loss, grad

def gradient_descent(x,y,w,a,iter):
    loss_vals=[]
    for i in range(0,iter):
        loss, grad=lossandgrad(x,y,w)
        
        w=w - (a*grad)
        loss_vals.append(loss)
    return loss_vals, w

def predictionfromprobability(x,w):
    probability= softmax_implement(x,w)
    prediction= np.argmax(probability, axis=1)
    return probability, prediction

from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
from metrics import false_positive_ratio
from sklearn.preprocessing import StandardScaler


def SoftmaxRegression(dataset, n, learning_rate, iterations):
    x_train, Y_train, x_test, Y_test= train_test_data (dataset, n)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(x_train)
    X_test = sc_X.fit_transform(x_test)
    w=np.random.rand(X_train.shape[1],len(np.unique(Y_train)))
    loss_vals, final_weight= gradient_descent(X_train,Y_train, w, learning_rate, iterations)
    plt.figure()
    plt.plot(loss_vals)
    probabilities, prediction=predictionfromprobability(X_train, final_weight)
    acc_train= accuracy(prediction, Y_train)
    test_prob, test_predict=predictionfromprobability(X_test, final_weight)
    acc_test = accuracy(test_predict, Y_test)
    confusion_mat= confusion_matrix(Y_train, prediction)
    pr_mat=precision_and_recall(confusion_mat)
    microf1= micro_average_f1_score(confusion_mat)
    macrof1= macro_average_f1_score(pr_mat)
    ratio= false_positive_ratio(confusion_mat)
    plt.show()

    return acc_train, acc_test, confusion_mat, pr_mat, microf1, macrof1, ratio


#testing purposes
a,b,c,d,e,f,g= SoftmaxRegression('train.csv',0.8, 0.001, 1000)
print (a)  #accuracy train

print(b)  #accuracy test



