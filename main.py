import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from SoftmaxRegression import SoftmaxRegression
from RandomForest import RandomForest


def main(ifname):
    accuracy_train=[]
    accuracy_test=[]
    microf1_scores=[]
    macrof1_scores=[]
    acc_RF=[]
    microf1_RF=[]
    macrof1_RF=[]
    nvals=[0.5,0.6,0.7,0.8,0.9]
    for n in nvals:
        acc_train, acc_test, confusion_mat, pr_mat, microf1, macrof1= SoftmaxRegression(ifname, n, 0.001, 
        iterations=3000, regularisation=1)
        acc, cm, prm, micro_f1, macro_f1, run_time = RandomForest(ifname, n, 40, 4)
        accuracy_train.append(acc_train)
        accuracy_test.append(acc_test)
        microf1_scores.append(microf1)
        macrof1_scores.append(macrof1)

        acc_RF.append(acc)
        microf1_RF.append(micro_f1)
        macrof1_RF.append(macro_f1)

    
    plt.figure()
    plt.grid()
    plt.subplot(211)
    plt.xlabel('Train-test splits')
    plt.plot(nvals, accuracy_train, label='Accuracy on train data (Logistic)', color='red')
    plt.plot(nvals, accuracy_test, label='Accuracy on test data (Logistic)', color ='red')
    plt.plot(nvals, acc_RF, label='Accuracy on test data (Random Forest)', color = 'green')
    plt.legend()
    plt.subplot(212)
    plt.xlabel('Train-test splits')
    plt.plot(nvals, microf1_scores, label='Micro average f1 score (Logistic)', color='red')
    plt.plot(nvals, macrof1_scores, label= 'Macro average f1 score (Logistic)', color='red')
    plt.plot(nvals, macrof1_RF, label= 'Macro average f1 score (Random Forest)', color='green')
    plt.plot(nvals, microf1_RF, label= 'Micro average f1 score (Random Forest)', color='green')
    plt.plot()
    plt.legend()
    plt.show()


    
if __name__ == '__main__':
    import sys
    main(sys.argv[1])




        