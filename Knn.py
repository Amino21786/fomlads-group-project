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
X_train, Y_train, X_test, Y_test = train_test_data('MobilePricingUpdated.csv', 0.8) #runs train_test_split function

# Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
list=[] #Create an empty list
i_range=range(1,103,2)
for i in i_range: # Give different odd values to k_neighbors
    start=time.process_time() 
           
    # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.        
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    y_pred_Knn = knn.predict(X_test)

    # I have tried the code with odd K-neighbors up until 103, to see if any changes occur. Happily, there are no changes after a certain threshold, but I have limited the values of k until 53, due to time constraints.
    # I have observed that for split=0.8, the minimum of k=25 to achieve accuracy 0.93(which is also the maximum value). For split=0.9, the minimum of k=7 to achieve 0.93 accuracy.
    
    # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))

    acc=accuracy(y_pred_Knn, Y_test) #computes accuracy
    cm=confusion_matrix(Y_test, y_pred_Knn) #computes test confusion matrix
    prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
    micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
    macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
    
    list.append(acc)
    print("-------------------------------------------------------------")
    print("Value of k for n_neighbors is ", i)
    print("Accuracy on Test set:", acc)
    print("Confusion Matrix:\n", cm)
    print("Precision and Recall Matrix: \n:", prm)
    print("Micro-average F1 Score:", micro_f1)
    print("Macro-average F1 Score:", macro_f1)
    
max_value=max(list)
max_index=list.index(max_value)
print("The maximum value of accuracy is ", max(list))
print("The index of the maximum accuracy is ", max_index)
print("The ideal k for best accuracy is  ", 2*max_index+1) # The step is 2, and we start the counting of indices from 0. So to find k, we multiply by 2 the index of maximum value of accuracy and then add 1.
print("List of accuracy is ", list)

plt.plot(i_range,list)
plt.xlabel('k')
plt.ylabel('Testing accuracy')
plt.title("Test_train_split with KNN ")
plt.show()


print("Time Taken:", time.process_time()-start) #time taken to compute all of it

   



# See if this helps to compare the classifiers and see which one performs best:

#classifier=[RandomForestClassifier(random_state=random_state),
#KNeighborsClassifier()]




'''
                                     Important observations (which should be polished more, but I am v tired and eyes are burning:((


I have performed the algorithm for all values of neighbors between 1 and 103, I have performed only for even values of neighbors, and also for odd values of neighbors.Each time, the accuracy has been lower for even values
than the one for an odd value of k.
See "a=0.8,evolution of k and accuracy.png"

Thus, I have chosen to work with odd values of k. The for loop for odd numbers is exactly the same for even numbers.
In the for loop for odd numbers, I have been trying to find the ideal k to train the data with.

 
I used different values of train test split=0.4,0.5,0.7,0.8,0.9,0.99, and different values of n_neighbors from 1->103 to see the evolution of the accuracy. The greatest accuracy which has been recorded is:

for test train split=0.4, the highest accuracy is 0.925833, which means that the ideal k is 23.
for test train split=0.5, the highest accuracy is 0.925, which means that the ideal k is 7.
for test train split=0.7, the highest accuracy is 0.93, which means that the ideal k is 21.
for test train split=0.8, the highest accuracy is 0.93, which means that the ideal k is 25.
for test train split=0.9, the highest accuracy is 0.93, which means that the ideal k is 7.
Also, for split=0.99, we can see that accuracy is 0.95 from the very first test, and it becomes 1 at the next iteration. This can easily lead to overfitting. As such, an optimal value for split should be 
less than or equal to 0.9.
we can find the ideal k by seeing what index our maximum value of accuracy has in the list, and then k is that value+1, because we start counting the indices from 0.

There are 6 plots on desktop, which show the evolution of accuracy for different k's in interval 1,51, and for different a. 


Since the RandomForest classifier has been using train test split=0.8, we shall use the same here to ensure that it remains fair.


I have decided to keep only odd values of k, because seeing that the maximum accuracy occurs only when k is odd, shows once again that an odd value of k works best for a data set with an even number of classes(as ours has 4).
Moreover, cutting back from the numbers of k cuts down the computation time significantly.

When taking the odd values up to 101, the tendancy for accuracy is to go down, hitting the lowest value of 0.9 for k=99.
Thus, we can take k=25, because this is the ideal k to reach maximum accuracy.
'''


