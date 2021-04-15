from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
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
import matplotlib.pyplot as plt

np.random.seed(42) #seed for repeatable results

def KNN():

    X_train, Y_train, X_test, Y_test = train_test_data('MobilePricingUpdated.csv', 0.8) #runs train_test_split function

    # Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
    list=[] #Create an empty list
    i_range=range(1,103,2) # Give different odd values to k_neighbors

    for i in i_range: 
        start=time.process_time() 
            
        # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
        knnClassifier=KNeighborsClassifier(n_neighbors=i)
        knnClassifier.fit(X_train,Y_train)
        y_pred_Knn = knnClassifier.predict(X_test)

        # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))

        acc=accuracy(y_pred_Knn, Y_test) #computes accuracy
        cm=confusion_matrix(Y_test, y_pred_Knn) #computes test confusion matrix
        prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
        micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
        macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
        
        list.append(acc)
        print("-------------------------------------------------------------")
        print ("Accuracy for KNN for k = ",i," is: ", acc) 
        print("Confusion Matrix:\n", cm)
        print("Precision and Recall Matrix: \n:", prm)
        print("Micro-average F1 Score:", micro_f1)
        print("Macro-average F1 Score:", macro_f1)
        
    max_value=max(list)
    print("The maximum value of accuracy is ", max(list), "and the ideal k to achieve this value is ", 2*list.index(max_value)+1)  # The step is 2, and we start the counting of indices from 0. 
    # So to find k, we multiply by 2 the index of maximum value of accuracy and then add 1.

    print("List of accuracy is ", list) 

    #Let us now plot the graph which shows the evolution of accuracy vs values of k from 1 up to and including 101.
    plt.plot(i_range,list)
    plt.xlabel('k')
    plt.ylabel('Testing accuracy')
    plt.title("Test_train_split with KNN ")
    plt.show()


    print("Time Taken:", time.process_time()-start) #time taken to compute all of it

KNN()

    



# See if this helps to compare the classifiers and see which one performs best:

#classifier=[RandomForestClassifier(random_state=random_state),
#KNeighborsClassifier()]




'''
                                     Important observations (which should be polished more, but I am v tired and eyes are burning:((

Since the RandomForest classifier has been using train test split=0.8, we shall use the same here to ensure that it remains fair.

I have performed the algorithm for all values of neighbors between 1 and 103. 

When performing for all values of k, I noticed that the accuracy of an even k was always at most equal with the value of the consecutive k (hence odd value).
For:

Even k from 2 to 102 (including), we have maximum accuracy = 0.925 , for the ideal k = 26, which takes time 1.39 to compute.
Odd k from 1 to 101 (including), we have maximum accuracy = 0.93 , for the ideal k = 25 ,which takes time 0.64 to compute.
All k from 1 to 101 (including), we have maximum accuracy = 0.93 , for the ideal k = 25 ,which takes time 1.18 to compute. (0.4375 without printing the list of accuracy)

 
I used different values of train test split=0.4,0.5,0.7,0.8,0.9,0.99, and different values of n_neighbors from 1->103 to see the evolution of the accuracy. The greatest accuracy which has been recorded is:

for test train split=0.4, the highest accuracy is 0.925833, which means that the ideal k is 23.
for test train split=0.5, the highest accuracy is 0.925, which means that the ideal k is 7.
for test train split=0.7, the highest accuracy is 0.93, which means that the ideal k is 21.
for test train split=0.8, the highest accuracy is 0.93, which means that the ideal k is 25.
for test train split=0.9, the highest accuracy is 0.93, which means that the ideal k is 7.
Also, for split=0.99, we can see that accuracy is 0.95 from the very first test, and it becomes 1 at the next iteration. This can easily lead to overfitting. As such, an optimal value for split should be 
less than or equal to 0.9.

As explaining in the code, we can find the ideal k by computing the index of the maximum value of accuracy in the list, and then k = 1 + index of the maximum value of list, because we start counting the indices from 0.



I have decided to keep only odd values of k, because seeing that the maximum accuracy occurs only when k is odd (for whichever test train split we use), shows once again that an odd value of k works best for a data set with an even number of classes
(as ours has 4). Moreover, cutting back from the numbers of k cuts down the computation time significantly.

When taking the odd values up to 301, the tendancy for accuracy is to go down, hitting the lowest value of 0.84 for k=301. (see picture in Knnplots file)



'''


