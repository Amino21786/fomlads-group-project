#This is all from Live session week 6 Google Collab
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Metrics analysis
#Confusion Matrix, Precision, Recall, False positive ratio
#All used to obtain the ROC curve

#measures how many y predictions match the y test set as a percentage
def accuracy(y_pred, y_test):
  return np.sum(y_pred == y_test)/y_test.size

def confusion_matrix(Y, Y_predict):
  K = len(np.unique(Y)) #how many unique elements in preidictions and ground truths (here it is 2)
  cm = np.zeros((K, K)) #4x4 matrix
  for i in range(len(Y)):
    cm[Y_predict[i]][Y[i]] += 1

  return cm

# The ratio of correct positive predictions to the total predicted positives (precision)
# The ratio of correct positive predictions to the total positives examples (recall)
def precision_and_recall(cm): #for 4x4 case
  p_r_matrix = np.zeros((4,2)) #this will have the 4 classes has rows and the 2 columns as the precision and recall for each class
  for i in range(4):
    recall = 0
    precision = 0
    fp = 0
    fn = 0
    tp = cm[i][i]
    for j in range(4):
        if i != j:
          fp += cm[i][j]
          fn += cm[j][i]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    p_r_matrix[i][0] = precision
    p_r_matrix[i][1] = recall
    
  return p_r_matrix # 4x2 matrix for the 4 classes and the precision and recall column for each class

      
# For the F1 score, since it is multi-class classification, we have to look at a different type of F1 score:
# Micro-Average F1 Score (looks at global precision and recall from the sums of TP. TN, FP, FN across the classes)
# This is a global F1 score which is known as the Micro-Average F1 Score (uses the same F1 score formula as binary classification)
def micro_average_f1_score(cm):
  fp = 0
  fn = 0
  tp = 0
  for i in range(4):
    tp += cm[i][i]
    for j in range(4):
        if i != j:
          fp += cm[i][j]
          fn += cm[j][i]
  global_precision = tp/(tp+fp)
  global_recall = tp/(tp+fn)

  return 2*((global_precision*global_recall)/(global_precision+global_recall)) #Micro-Average F1 Score



#Second type of F1 Score is more helpful when there are imbalanced classes, it takes an average of each precision and recall values for each classes and uses these as the precision and recall
#values and then use the standard F1 score equation
def macro_average_f1_score(p_r_matrix):
  sum_precision = 0
  sum_recall = 0
  for i in range(4):
    sum_precision += p_r_matrix[i][0]
    sum_recall += p_r_matrix[i][1]

  average_total_precision = sum_precision/4
  average_total_recall = sum_recall/4

  return 2*((average_total_precision*average_total_recall)/(average_total_precision+average_total_recall)) #Macro-Average F1 Score



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

