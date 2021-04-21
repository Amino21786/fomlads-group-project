import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from SoftmaxRegression import SoftmaxRegression
from sklearn.ensemble import RandomForestClassifier
from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")

def rf_hyperparameters(dataset):
    X_train, Y_train, X_test, Y_test = train_test_data(dataset, 0.8) #runs train_test_split function
    n_range=range(10, 100)
    acc_scores=[]
    mi_f1=[]
    ma_f1=[]
    for i in n_range:
        RFClassifier = RandomForestClassifier(n_estimators=i, random_state=4)
        RFClassifier.fit(X_train,Y_train)
        y_pred_RF = RFClassifier.predict(X_test)
        acc = accuracy(y_pred_RF, Y_test) 
        cm = confusion_matrix(Y_test, y_pred_RF) 
        prm = precision_and_recall(cm) 
        micro_f1 = micro_average_f1_score(cm)
        macro_f1 = macro_average_f1_score(prm) 
        acc_scores.append(acc)
        mi_f1.append(micro_f1)
        ma_f1.append(macro_f1)

    plt.plot(n_range, acc_scores, label='Accuracy on test data')
    plt.title('Test Accuracy vs Number of trees')
    plt.xlabel('Number of trees (n_estimators)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('Random Forest Accuracy Graph')
    plt.close()

    plt.plot(n_range, mi_f1, label='Micro F1 Score')
    plt.plot(n_range, ma_f1, label='Macro F1 Score')
    plt.title('F1 Score vs Number of trees')
    plt.xlabel('Number of trees (n_estimators)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('Random Forest F1 Score Graph')
    plt.close()
    
    

def oob_error_rf(dataset):
    X_train, Y_train, X_test, Y_test = train_test_data(dataset, 0.8) #runs train_test_split function
    n_range=range(15,100)
    oob_errors=[]
    for i in n_range:
        RFClassifier = RandomForestClassifier(n_estimators=i, oob_score=True, random_state=4)
        RFClassifier.fit(X_train,Y_train)
        oob_errors.append(1-RFClassifier.oob_score_)

    plt.plot(n_range, oob_errors)
    plt.title('OOB error vs Number of trees')
    plt.xlabel('Number of trees (n_estimators)')
    plt.ylabel('OOB Error')
    plt.savefig('Random Forest Loss Function Graph')
    plt.close()
    
 
# not scaled, below:

# n is the test_train_split value below:
def Knn_hyperparameters(dataset):
    X_train, Y_train, X_test, Y_test = train_test_data(dataset,0.8) #runs train_test_split function
    # Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
    #Create some empy lists which will be used for plots
    acc_scores_Knn=[]
    mi_f1_Knn=[]
    ma_f1_Knn=[]
    # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
    for i in range(1,103,2):
        knnClassifier=KNeighborsClassifier(n_neighbors=i)
        knnClassifier.fit(X_train,Y_train)
        y_pred_Knn = knnClassifier.predict(X_test)
        # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
        acc_Knn=accuracy(y_pred_Knn, Y_test) #computes accuracy
        cm_Knn=confusion_matrix(Y_test, y_pred_Knn) #computes test confusion matrix
        prm_Knn = precision_and_recall(cm_Knn) #computes precision and recall and represents the in a matrix
        micro_f1_Knn = micro_average_f1_score(cm_Knn) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
        macro_f1_Knn = macro_average_f1_score(prm_Knn) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
        acc_scores_Knn.append(acc_Knn)
        mi_f1_Knn.append(micro_f1_Knn)
        ma_f1_Knn.append(macro_f1_Knn)

    plt.plot(range(1,103,2),acc_scores_Knn,label='Accuracy on test for Knn')
    plt.title('Test Accuracy vs K neighbors')
    plt.xlabel('Number of neighbors (k neighbors)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('Knn Accuracy Graph')
    plt.close()

    plt.plot(range(1,103,2),mi_f1_Knn,label='Micro f1 score for Knn')
    plt.plot(range(1,103,2), ma_f1_Knn, label='Macro F1 Score')
    plt.title('F1 Score vs K neighbors')
    plt.xlabel('Number of neighbors (k neighbors)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('Knn F1 Score Graph')
    plt.close()
    


# Error rate
    
def error_function_Knn(dataset):
    error_rate=[]
    X_train, Y_train, X_test, Y_test = train_test_data(dataset,0.8) #runs train_test_split function
    for i in range(1,103,2):
        knnClassifier=KNeighborsClassifier(n_neighbors=i)
        knnClassifier.fit(X_train,Y_train)
        y_pred_Knn = knnClassifier.predict(X_test)
        error_rate.append(np.mean(y_pred_Knn!=Y_test))
    
    plt.plot(range(1,103,2),error_rate,color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=5)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K neighbors')
    plt.ylabel('Error Rate')
    plt.savefig('Knn Error function')
    plt.close()



''' These will be saved when running main.py, so there is no need to run them here.

Knn_hyperparameters('MobilePricingUpdated.csv')
error_function_Knn('MobilePricingUpdated.csv')
rf_hyperparameters('MobilePricingUpdated.csv')
oob_error_rf('MobilePricingUpdated.csv')

'''




