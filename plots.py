import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SoftmaxRegression import SoftmaxRegression












#tried to use the code from the main py as a function but stuck, for now will have this plotting idea as it is in the main file and for the separate model plots can have the in this file
"""
def accuracy_graph(model):
    nvals=[0.5,0.6,0.7,0.8,0.9]
    models=['Random Forest', 'KNN']
    fig=plt.figure()
    plt.xlabel('Train-test splits')
    plt.title('Accuracy vs Train-test splits')
    fig_2=plt.figure()
    plt.xlabel('Train-test splits')
    plt.title('F1 Score vs Train-test splits')

    for i in model:
        accuracy_test=[]
        microf1_scores=[]
        macrof1_scores=[]
        for n in nvals:
            acc_test, confusion_mat, pr_mat, microf1, macrof1, run_time = model[i]
            accuracy_test.append(acc_test)
            microf1_scores.append(microf1)
            macrof1_scores.append(macrof1)
        fig += plt.plot(nvals, acc_test, label='Accuracy on test data ('+models[i]+')')
        plt.legend()
        
        plt.plot(nvals, microf1_scores, label='Micro average f1 score ('+models[i]+')')
        plt.plot(nvals, macrof1_scores, label= 'Macro average f1 score ('+models[i]+')')
        plt.plot()
        plt.legend()
   
"""