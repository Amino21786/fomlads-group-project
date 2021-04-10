#This is all from Live session week 6 Google Collab
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Metrics analysis
#Confusion Matrix, Precision, Recall, False positive ratio
#All used to obtain the ROC curve

#measures how many y predictions match the y test set as a percentage
def accuracy(y_pred, y_test):
  return np.sum(y_pred==y_test)/y_test.size

def confusion_matrix(Y, Y_predict):
  K = len(np.unique(Y)) #how many unique elements in preidictions and ground truths (here it is 2)
  cm = np.zeros((K, K)) #4x4 matrix
  for i in range(len(Y)):
    cm[Y_predict[i]][Y[i]]+=1

  return cm

# The ratio of correct positive predictions to the total predicted positives
def precision_and_recall(cm): #for 4x4 case
  for i in range(4):
    recall=0
    precision=0
    fp=0
    fn=0
    tp=cm[i][i]
    for j in range(4):
        if i != j:
          fp+=cm[i][j]
          fn+=cm[j][i]
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    print("Precision for Price Range ", i, " is ",precision)
    print("Recall for Price Range ", i , " is ", recall)

      
# The ratio of correct positive predictions to the total positives examples
def recall(cm):
  return cm[1][1]/(cm[1][1] + cm[1][0])
  
# The probability of falsely rejecting the null hypothesis for a particular test
# The false positive rate is calculated as the ratio between the number of false positives
# and the total number of actual negative events (regardless of classification)
def false_positive_ratio(cm):
  return cm[0][1]/(cm[0][1] + cm[0][0])




#Everything below this can go in the individual model files
"""
# Define decision thresholds between 0-1
thresholds = np.linspace(0,1, 400)

# Calculate the recall and false positive rate for all the predefined threshold options
def get_roc(x, y):
  tpr = []
  fpr = []
  for threshold in thresholds:
    Y_predict = classifier.predict(x, threshold=threshold) 
    cm = confusion_matrix(y, Y_predict)
    tpr.append(recall(cm))
    fpr.append(false_positive_ratio(cm))
  return fpr, tpr
  
# Get the best cutoff point determining the best threshold and intuitively we want to maximise the true positive rate (recall) and minimise the false positive rate
def get_cutoff(fpr, tpr):
  optimal_idx = np.argmax(np.array(tpr) - np.array(fpr))
  optimal_threshold = thresholds[optimal_idx]
  return optimal_idx, optimal_threshold


train_roc = get_roc(X_train, Y_train)
test_roc = get_roc(X_test, Y_test)

train_cutoff = get_cutoff(train_roc[0], train_roc[1])
test_cutoff = get_cutoff(test_roc[0], test_roc[1])

plt.plot(train_roc[0], train_roc[1], label="Train", c='r')
plt.plot(test_roc[0], test_roc[1], label="Test", c='g', linestyle='dashed')
plt.scatter(train_roc[0][train_cutoff[0]], train_roc[1][train_cutoff[0]], label="Train cutoff: {}".format(train_cutoff[1]), c='r')
plt.scatter(test_roc[0][test_cutoff[0]], test_roc[1][test_cutoff[0]], label="Test cutoff: {}".format(test_cutoff[1]), c='g', linestyle='dashed')

plt.title("ROC Curve")
plt.xlabel("False positive ratio")
plt.ylabel("Recall")
plt.legend()
"""
#the curve tells us how well our classifier works on our data, given the training and test sets
#we want to maximise the area
#cutoff point will maximises our true positive rate (recall) and minimise the the false positive rate

