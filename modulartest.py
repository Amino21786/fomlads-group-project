from sklearn.ensemble import RandomForestClassifier
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
df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
xs=df.drop(['price_range'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['price_range'].to_numpy() #The target variable

#forming the train and test data split, through index arrays
index_array=np.arange(len(ys))
train_index_array = index_array[:int(len(index_array) * 0.8)]
test_index_array = index_array[int(len(index_array) * 0.8):]
# get train and test subsets
x_train = xs[train_index_array]
y_train = ys[train_index_array]
x_test = xs[test_index_array]
y_test = ys[test_index_array]

# Start to implement the Random Forest Classifier
for i in [40,80,90,100]:
    for j in [1, 2, 4]:
        start=time.process_time()
        RFClassifier = RandomForestClassifier(n_estimators=i, random_state=j)
        RFClassifier.fit(x_train,y_train)
        y_pred_RF = RFClassifier.predict(x_test)
        print("n_estimator, random_state: ", i, j)

        #5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
        acc = accuracy(y_pred_RF, y_test) #computes accuracy
        cm = confusion_matrix(y_test, y_pred_RF) #computes test confusion matrix
        prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
        micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
        macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
        print("Accuracy on Test set:", acc)
        print("Confusion Matrix:\n", cm)
        print("Precision and Recall Matrix: \n:", prm)
        print("Micro-average F1 Score:", micro_f1)
        print("Macro-average F1 Score:", macro_f1)
        print("Time Taken:", time.process_time()-start) #time taken to compute all of it


