from sklearn.ensemble import RandomForestClassifier
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision
from metrics import recall
from metrics import false_positive_ratio
import pandas as pd
import numpy as np
import time



np.random.seed(42) #seed for repeatable results
df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
xs=df.drop(['price_range'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['price_range'] #The target variable

#forming the train and test data split, through index arrays
index_array=np.arange(len(ys))
i=[0.5,0.7,0.75,0.8,0.85,0.9,0.95,0.99] #train test split takes different values, was previously : train_test_split = 0.7
for train_test_split in i:
    train_index_array = index_array[:int(len(index_array) * train_test_split)]
    test_index_array = index_array[int(len(index_array) * train_test_split):]
# get train and test subsets
    x_train = xs[train_index_array]
    y_train = ys[train_index_array]
    x_test = xs[test_index_array]
    y_test = ys[test_index_array]
    start=time.process_time()
# Start to implement the Random Forest Classifier
    RFClassifier = RandomForestClassifier(n_estimators=50)
    RFClassifier.fit(x_train,y_train)
    y_pred_RF = RFClassifier.predict(x_test)
    acc=accuracy(y_pred_RF, y_test) #compute accuracy
    print("Accuracy on Test set for train_test_split=",train_test_split,"is", acc)
    print("Confusion Matrix:", confusion_matrix(y_test,)
    print("Time Taken:", time.process_time()-start)

# We have played around with the values for train_test_split and when using the 0.7 value, we see the following:
# The Accuracy on Test with Ranndom Forest is 0.8474723864521747 and it takes roughly 3.5-4 seconds at most to compute this(for the entire data-set, depending on your machine).
# Later, we thought we can write a for loop which takes different values of train_test_split and proceed to doing the same thing as above. The values for accuracy vary, 
# the lowest being  0.8333974975938402 for train_test_split=0.5 and the greatest is 0.9429553264604811 for train_test_split=0.99. 
# However, we can see that for train_test_split values increasing by 0.1, the accuracy values fluctuate between a starting value for 0.7, then decreasing for 0.75,and increasing at a later point for 0.8.

# We will need the time at a later point in the report, where we shall compare the time needed for each method to complete the same test.