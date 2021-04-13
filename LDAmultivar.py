import numpy as np 
import pandas as pd 
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from modelconstruct import train_test_data
from metrics import accuracy
from metrics import confusion_matrix
from metrics import precision_and_recall
from metrics import micro_average_f1_score
from metrics import macro_average_f1_score

np.random.seed(42) #seed for repeatable results
X_train, Y_train, X_test, Y_test = train_test_data('MobilePricingUpdated.csv', 0.8) #runs train_test_split function

start=time.process_time()
lda=LinearDiscriminantAnalysis()
lda = lda.fit(X_train,Y_train)
y_pred_LDA = lda.predict(X_test)


#5 metrics (Accuracy, confusion Matrix, precision, recall and f1 (both types))
acc = accuracy(y_pred_LDA, Y_test) #computes accuracy
cm = confusion_matrix(Y_test, y_pred_LDA) #computes test confusion matrix
prm = precision_and_recall(cm) #computes precision and recall and represents the in a matrix
micro_f1 = micro_average_f1_score(cm) #Micro f1, uses the global precision and recall (prone to imbalanced datasets)
macro_f1 = macro_average_f1_score(prm) #Macro f1, uses the average of individual classes' precision and recalls (better for imbalanced datasets)
print("Accuracy on Test set:", acc)
print("Confusion Matrix:\n", cm)
print("Precision and Recall Matrix: \n:", prm)
print("Micro-average F1 Score:", micro_f1)
print("Macro-average F1 Score:", macro_f1)
print("Time Taken:", time.process_time()-start) #time taken to compute all of it