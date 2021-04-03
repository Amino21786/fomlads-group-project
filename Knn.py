from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import warnings
import timeit
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


start=time.process_time()

np.random.seed(42) #seed for repeatable results
df=pd.read_csv('RainAUS.csv') #Importing the preprocessed dataset
xs=df.drop(['Date', 'Location', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['RainTomorrow'].to_numpy() #The target variable

#forming the train and test data split, through index arrays
index_array=np.arange(len(ys)) # 0.75,0.8,0.85,0.9,0.95,0.99
i=[0.5,0.7,0.75,0.8,0.85,0.9,0.95,0.99] #train test split takes different values, was previously : train_test_split = 0.7
for train_test_split in i:
    train_index_array = index_array[:int(len(index_array) * train_test_split)]
    test_index_array = index_array[int(len(index_array) * train_test_split):]
# get train and test subsets
    x_train = xs[train_index_array]
    y_train = ys[train_index_array]
    x_test = xs[test_index_array]
    y_test = ys[test_index_array]
# Start to implement the Knn, using the same data for train_split_test as for Random Forest classifier in order to make the tests as fair as possible.
    knn=KNeighborsClassifier()
    knn.fit(x_train,y_train)
    knn_head=knn.predict(x_test)
    print(f"""accuracy_score: {accuracy_score(knn_head, y_test)}
    roc_auc_score: {roc_auc_score(knn_head, y_test)}""")



# accuracy_score: 0.8053989642055089
# roc_auc_score: 0.7180164761673166
#result = timeit.timeit("'-'.join([str(i) for i in range(10000)])", number=100000)
#print(result)
""" for i=[0.5,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
 accuracy_score: 0.8203079884504331
#    roc_auc_score: 0.7503364938566903
##accuracy_score: 0.8363582199000871
 #   roc_auc_score: 0.7528976156699123
#accuracy_score: 0.8305788532930015
#    roc_auc_score: 0.7532734909554242
accuracy_score: 0.8433246253265503
    roc_auc_score: 0.7605985660832587
accuracy_score: 0.8386268848251524
    roc_auc_score: 0.7505004404718556
accuracy_score: 0.848893166506256
    roc_auc_score: 0.7486783803283108
accuracy_score: 0.8631926302763646
    roc_auc_score: 0.7708477215099538
accuracy_score: 0.9415807560137457
    roc_auc_score: 0.8211688311688311
213.890625- time
"""


print(time.process_time()-start)