import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Metrics analysis
#Confusion Matrix, Precision, Recall, F1 Score (Micro and Macro)


#measures how many y predictions match the y test set as a percentage
def accuracy(y_pred, y_test):
  return np.sum(y_pred == y_test)/y_test.size

# N X N matrix of the true positives, true negatives, false positives and false negatives of the dataset, in our case it is a 4 x 4 matrix
def confusion_matrix(Y, Y_predict):
  K = len(np.unique(Y)) #how many unique elements in preidictions and ground truths (here it is 4)
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





