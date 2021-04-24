from sklearn.neighbors import KNeighborsClassifier
from modelconstruct import train_test_data
from modelconstruct import standardscaler # Importing this for the scaled data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
np.random.seed(42) #seed for repeatable results
plt.style.use('seaborn-whitegrid')



# Code for UNSCALED data:

# n is the test_train_split value below:
def Knn(dataset,n,Kneighbors):
    X_train, Y_train, X_test, Y_test = train_test_data(dataset,n) #runs train_test_split function
    start=time.process_time() 
    # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
    knnClassifier=KNeighborsClassifier(n_neighbors=Kneighbors)
    knnClassifier.fit(X_train,Y_train)
    y_pred_Knn = knnClassifier.predict(X_test)

    # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
    acc_Knn=accuracy(y_pred_Knn, Y_test) #computes accuracy
    cm_Knn=confusion_matrix(Y_test, y_pred_Knn) #computes test confusion matrix
    prm_Knn = precision_and_recall(cm_Knn) #computes precision and recall and represents them in a matrix
    micro_f1_Knn = micro_average_f1_score(cm_Knn) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
    macro_f1_Knn = macro_average_f1_score(prm_Knn) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
    run_time_Knn=time.process_time()-start #time taken to compute all of it

    return acc_Knn, micro_f1_Knn, macro_f1_Knn, run_time_Knn

# Train test split can take different values. However, we will choose a=0.8, to make the test as fair as possible.





# Code for SCALED data

def Knn_scaled(dataset,n,Kneighbors):
    x_train, Y_train, x_test, Y_test = train_test_data(dataset,n) #runs train_test_split function
    # Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
    start=time.process_time() 
    X_train=standardscaler(x_train)
    X_test = standardscaler(x_test)
    # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
    knnClassifier=KNeighborsClassifier(n_neighbors=Kneighbors)
    knnClassifier.fit(X_train,Y_train)
    y_pred_Knn = knnClassifier.predict(X_test)

    # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
    acc_Knn_scaled=accuracy(y_pred_Knn, Y_test) #computes accuracy
    cm_Knn_scaled=confusion_matrix(Y_test, y_pred_Knn) #computes test confusion matrix
    prm_Knn_scaled = precision_and_recall(cm_Knn_scaled) #computes precision and recall and represents them in a matrix
    micro_f1_Knn_scaled = micro_average_f1_score(cm_Knn_scaled) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
    macro_f1_Knn_scaled = macro_average_f1_score(prm_Knn_scaled) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
    run_time_Knn_scaled=time.process_time()-start #time taken to compute all of it

    return acc_Knn_scaled, micro_f1_Knn_scaled, macro_f1_Knn_scaled, run_time_Knn_scaled



#Finding the ideal k for test_train_split=0.8:

def ideal_k(dataset):
    # Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
    X_train, Y_train, X_test, Y_test = train_test_data(dataset,0.8) 
    listacc=[] #Create an empty list
    i_range=range(1,103,2) # Give different odd values to k_neighbors
    for i in i_range: 
        # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
        knnClassifier=KNeighborsClassifier(n_neighbors=i)
        knnClassifier.fit(X_train,Y_train)
        y_pred_Knn = knnClassifier.predict(X_test)
        # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
        # We have used accuracy as a very important metric to measure how capable a model is for classification
        acc=accuracy(y_pred_Knn, Y_test) #computes accuracy
        listacc.append(acc)
    max_value=max(listacc)
    print('---------------------------------------------')
    print("For Knn for unscaled data: The maximum value of accuracy is ", max(listacc), "and the ideal k to achieve this value is ", 2*listacc.index(max_value)+1)  
    # The step is 2, and we start the counting of indices from 0. So, to find k, we multiply by 2 the index of maximum value of accuracy and then add 1.
    
print("---------------------------------------------------------------------")
print('Knn for unscaled data: The accuracy, micro_f1, macro_f1 and the run_time are the following')
print(Knn('MobilePricingUpdated.csv',0.8,25))
print('Knn for scaled data: The accuracy, micro_f1, macro_f1 and the run_time are the following')
print(Knn_scaled('MobilePricingUpdated.csv',0.8,77))

#For the scaled data, the ideal k is 77 and it gives greatest accuracy of 0.665
