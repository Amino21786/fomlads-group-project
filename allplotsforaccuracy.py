from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
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
df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
xs=df.drop(['price_range'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['price_range'].to_numpy() #The target variable

#forming the train and test data split, through index arrays
index_array=np.arange(len(ys)) 

# Train test split can take different values. However, we will choose a=0.8 as in RandomForest.py, to make the test as fair as possible.
start=time.process_time()
k_range=range(1,51)
for a in [0.4,0.5,0.7,0.8,0.9]: # Give different values to train test split
    list=[] #Create an empty list
    print("------------------------------------------------------")
    print("Value of test train split is: ",a)
    train_index_array = index_array[:int(len(index_array) * a)]
    test_index_array = index_array[int(len(index_array) * a):]
        # get train and test subsets
    x_train = xs[train_index_array]
    y_train = ys[train_index_array]
    x_test = xs[test_index_array]
    y_test = ys[test_index_array]
            
    for i in k_range: # Give different odd values to k_neighbors
           
            # Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.
            
            knn=KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train,y_train)
            y_pred_Knn = knn.predict(x_test)

            # I have tried the code with odd K-neighbors up until 105, to see if any changes occur. Happily, there are no changes after a certain threshold, but I have limited the values of k until 53, due to time constraints.
            # I have observed that for split=0.8, the minimum of k=25 to achieve accuracy 0.93(which is also the maximum value). For split=0.9, the minimum of k=7 to achieve 0.93 accuracy.
            # Now we shall compute 5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))

            acc=accuracy(y_pred_Knn, y_test) #computes accuracy
            cm=confusion_matrix(y_test, y_pred_Knn) #computes test confusion matrix
            prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
            micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
            macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
            
            list.append(acc)
            """
            
            print("Accuracy on Test set:", acc)
            print("Confusion Matrix:\n", cm)
            print("Precision and Recall Matrix: \n:", prm)
            print("Micro-average F1 Score:", micro_f1)
            print("Macro-average F1 Score:", macro_f1)

            """
            
    max_value=max(list)
    max_index=list.index(max_value)
    print("The maximum value of accuracy is ", max(list))
    print("The index of the maximum accuracy is ", max_index)
    print("The ideal k for best accuracy is  ", max_index+1)
    print("List of accuracy is ", list)

    plt.plot(k_range,list)
    plt.xlabel('k')
    plt.ylabel('Testing accuracy')
    plt.title("Each plot is for test train split = 0.7,0.8,0.9 ")
    plt.show()

    

print("Time Taken:", time.process_time()-start) #time taken to compute all of it
   
   
'''
        list_even=[] #Create an empty list
        for j in range(4,54,2): #I have tried the code with K-neighbors up until 105, to see if any changes occur. Happily, there are no changes after a certain threshold, but I have limited the values of k until 53, due to time constraints.
            # I have observed that for split=0.8, the minimum of k=25 to achieve accuracy 0.93(which is also the maximum value). For split=0.9, the minimum of k=7 to achieve 0.93 accuracy.
            #print("-----")
            #print("n_neighbors=",h)
            #print("-----")

            knn=KNeighborsClassifier(j)
            knn.fit(x_train,y_train)
            knn.score(x_test,y_test)
            y_pred_Knn = knn.predict(x_test)

            
            #print("Accuracy:",metrics.accuracy_score(y_test,y_pred_Knn))
            acc=accuracy(y_pred_Knn, y_test) #computes accuracy
            #cm=confusion_matrix(y_test, y_pred_Knn) #computes test confusion matrix
            #print("Accuracy on Test set:", acc)
            #print("Confusion Matrix:\n", cm)
            #print(precision_and_recall(cm))
            #print("------------------------------------------------------")
            list_even.append(acc)
        print("List of accuracy for even neighbors is ", list_even)
        max_value_even=max(list_even)
        max_index_even=list.index(max_value_even)
        print("The maximum value of accuracy is ", max(list_even))
        print("The index of the maximum accuracy is ", max_index_even)
        #print("The accuracy is max for k=",)




'''


# See if this helps:
#classifier=[RandomForestClassifier(random_state=random_state),
#KNeighborsClassifier()]




'''
Important observations:

 We can change the value of n_neighbors=4/5 or ahy other number as a matter of fact.
we usually take k to be an odd number. When we set k=4, we can see that the accuracy is worse than for k=5, for instance.
For results, see pictures on desktop: for neighbors=4,see kneighbors=4.jpg.
 for neighbors=5, see kneighbors=5.jpg.
 for neighbors=51, see kneighbors=51.jpg.

 I have performed the algorithm for even values of neighbors, and each time, the accuracy has been lower than the one for the odd value of k.
 See "odd vs even k.jpg"
 Thus, I have chosen to work with odd values of k, as data scientists do anyway. The for loop for odd numbers is exactly the same for even numbers.
 In the for loop for odd numbers, I have been trying to find the ideal k to train the data with.

 Multiple trials have shown that for neighbors =51, the accuracy is below the one for neighbors=31. As such, 
 we'll try to find the optimal value which gives the highest accuracy.

 Also, for split=0.99, we can see that accuracy is 0.95 from the very first test, and from the next one, it becomes 1. This can easily lead to overfitting.
 As such, an optimal value for split should be less than or equal to 0.9. 



I have been running the code, for different values of train test split=0.7,0.8,0.9, different values of n_neighbors from 1->51
to see the evolution of the accuracy. The greatest accuracy which has been recorded is:


for test train split=0.4, the highest accuracy is 0.925833, which means that the ideal k is 23.
for test train split=0.5, the highest accuracy is 0.925, which means that the ideal k is 7.
for test train split=0.7, the highest accuracy is 0.93, which means that the ideal k is 21.
for test train split=0.8, the highest accuracy is 0.93, which means that the ideal k is 25.
for test train split=0.9, the highest accuracy is 0.93, which means that the ideal k is 7.

we can find the ideal k by seeing what index our maximum value of accuracy has in the list, and then k is that value+1, because
 we start counting the indices from 0.
 
 There are 6 plots on desktop, which show the evolution of accuracy for different k's in interval 1,51, and for different a. 
 Since the RandomForest classifier has been using train test split=0.8, we shall use the same here to ensure that it remains fair.

'''


