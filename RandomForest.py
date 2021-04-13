from sklearn.ensemble import RandomForestClassifier
from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
from metrics import false_positive_ratio
import pandas as pd
import numpy as np
import time

np.random.seed(42) #seed for repeatable results


def RandomForest():
    X_train, Y_train, X_test, Y_test = train_test_data('MobilePricingUpdated.csv', 0.8) #runs train_test_split function
    for i in [40,80,90,100]:
        for j in [1, 2, 4]:
            start=time.process_time()
            RFClassifier = RandomForestClassifier(n_estimators=i, random_state=j)
            RFClassifier.fit(X_train,Y_train)
            y_pred_RF = RFClassifier.predict(X_test)
            print("n_estimator, random_state: ", i, j)

            #5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
            acc = accuracy(y_pred_RF, Y_test) #computes accuracy
            cm = confusion_matrix(Y_test, y_pred_RF) #computes test confusion matrix
            prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
            micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
            macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
            print("Accuracy on Test set:", acc)
            print("Confusion Matrix:\n", cm)
            print("Precision and Recall Matrix: \n:", prm)
            print("Micro-average F1 Score:", micro_f1)
            print("Macro-average F1 Score:", macro_f1)
            print("Time Taken:", time.process_time()-start) #time taken to compute all of it

RandomForest()


# So I tested the hyperparameters, settled on train test split for 0.8
#for the RF hyperparameters, n_estimators and random_state, iterated from 10-100 (in 10 intervals for n_estimators) and None - 4 (5 intervals for random_state)
#Best combination from the 50 trials were n=40 and random_state=4 (best accuracy=0.8825, F1 scores in 0.882-0.883), next one was n=80, random_state=1 (with around 0.880 for f1 and accuracy)
#Worst combination was n=10 and no random state with around 0.75 for the metrics