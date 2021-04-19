import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from SoftmaxRegression import SoftmaxRegression
from sklearn.ensemble import RandomForestClassifier
from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score




def RF_graph():
    X_train, Y_train, X_test, Y_test = train_test_data('MobilePricingUpdated.csv', 0.8) #runs train_test_split function
    n_range=range(10, 100, 10)
    acc_scores=[]
    mi_f1=[]
    ma_f1=[]
    for i in n_range:
        start=time.process_time()
        RFClassifier = RandomForestClassifier(n_estimators=i, random_state=4)
        RFClassifier.fit(X_train,Y_train)
        y_pred_RF = RFClassifier.predict(X_test)
        #5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
        acc = accuracy(y_pred_RF, Y_test) #computes accuracy
        cm = confusion_matrix(Y_test, y_pred_RF) #computes test confusion matrix
        prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
        micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
        macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
        acc_scores.append(acc)
        mi_f1.append(micro_f1)
        ma_f1.append(macro_f1)
        #print("Time Taken:", time.process_time()-start) #time taken to compute all of it
    plt.figure()
    plt.grid()
    plt.xlabel('Number of trees (n_estimators)')
    plt.ylabel('Accuracy')
    plt.plot(n_range, acc_scores, label='Accuracy on test data')
    plt.plot(n_range, mi_f1, label='Micro F1 Score')
    plt.plot(n_range, ma_f1, label='Macro F1 Score')
    plt.legend()
    plt.show()

RandomForest()







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