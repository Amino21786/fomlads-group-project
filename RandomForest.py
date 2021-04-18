from sklearn.ensemble import RandomForestClassifier
from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(42) #seed for repeatable results


def RandomForest(dataset, n, n_trees, ran_state):
    X_train, Y_train, X_test, Y_test = train_test_data(dataset, n) #runs train_test_split function
    start=time.process_time()
    RFClassifier = RandomForestClassifier(n_estimators=n_trees, random_state=ran_state)
    RFClassifier.fit(X_train,Y_train)
    y_pred_RF = RFClassifier.predict(X_test)
    #5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
    acc = accuracy(y_pred_RF, Y_test) #computes accuracy
    cm = confusion_matrix(Y_test, y_pred_RF) #computes test confusion matrix
    prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
    micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
    macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
    run_time = time.process_time()-start #time taken to compute all of it

    return acc, cm, prm, micro_f1, macro_f1, run_time

acc= RandomForest('MobilePricingUpdated.csv', 0.8, 40, 4)


# So I tested the hyperparameters, settled on train test split for 0.8
#for the RF hyperparameters, n_estimators and random_state, iterated from 10-100 (in 10 intervals for n_estimators) and None - 4 (5 intervals for random_state)
#Best combination from the 50 trials were n=40 and random_state=4 (best accuracy=0.8825, F1 scores in 0.882-0.883), next one was n=80, random_state=1 (with around 0.880 for f1 and accuracy)
#Worst combination was n=10 and no random state with around 0.75 for the metrics

"""
old file
def RandomForest():
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
"""