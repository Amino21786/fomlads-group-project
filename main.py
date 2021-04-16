import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse
from mainSoftmaxRegression import SoftmaxRegression
def main(ifname):
    accuracy_train=[]
    accuracy_test=[]
    microf1_scores=[]
    macrof1_scores=[]
    falsepositiveratios=[]
    nvals=[0.5,0.6,0.7,0.8,0.9]
    for n in nvals:
        acc_train, acc_test, confusion_mat, pr_mat, microf1, macrof1, ratio= SoftmaxRegression(ifname, n, 0.001, 
        iterations=3000, regularisation=1)
        accuracy_train.append(acc_train)
        accuracy_test.append(acc_test)
        microf1_scores.append(microf1)
        macrof1_scores.append(macrof1)


    plt.figure()
    plt.subplot(211)
    plt.xlabel('Train-test splits')
    plt.plot(nvals, accuracy_train, label='Accuracy on train data')
    plt.plot(nvals, accuracy_test, label='Accuracy on test data')
    plt.legend()
    plt.subplot(212)
    plt.xlabel('Train-test splits')
    plt.plot(nvals, microf1_scores, label='Micro average f1 score')
    plt.plot(nvals, macrof1_scores, label= 'Macro average f1 score')
    plt.plot()
    plt.legend()
    plt.show()

    
if __name__ == '__main__':
    import sys
    main(sys.argv[1])




        