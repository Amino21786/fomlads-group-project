from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
np.random.seed(42) #seed for repeatable results
df=pd.read_csv('RainAUS.csv') #Importing the preprocessed dataset
xs=df.drop(['Date', 'Location', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['RainTomorrow'].to_numpy() #The target variable

#forming the train and test data split, through index arrays
index_array=np.arange(len(ys))
train_test_split = 0.7
train_index_array = index_array[:int(len(index_array) * train_test_split)]
test_index_array = index_array[int(len(index_array) * train_test_split):]
# get train and test subsets
x_train = xs[train_index_array]
y_train = ys[train_index_array]
x_test = xs[test_index_array]
y_test = ys[test_index_array]
RFClassifier = RandomForestClassifier(n_estimators = 20,random_state = 1)
RFClassifier.fit(x_train,y_train)
y_pred_RF = RFClassifier.predict(x_test)
acc=np.sum(y_pred_RF==y_test)/y_test.size
print("Accuracy on Test set:", acc)
0.8456964113845731