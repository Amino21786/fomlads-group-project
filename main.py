import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from data import correlation_heatmap
from SoftmaxRegression import SoftmaxRegression
from RandomForest import RandomForest
from Knn import Knn
from Knn import Knn_scaled
from LDA import LDA
from plots import rf_hyperparameters, oob_error_rf, Knn_hyperparameters, error_function_Knn
from Knn import ideal_k_scaled, ideal_k_unscaled




def main(ifname):
    plt.style.use('seaborn-whitegrid')
    
    #Correlation Heatmap
    correlation_heatmap(ifname)
    #Random Forest graphs:
    rf_hyperparameters(ifname)
    oob_error_rf(ifname)
    #Knn graphs and prints ideal k:
    ideal_k_scaled(ifname)
    ideal_k_unscaled(ifname)
    Knn_hyperparameters(ifname)
    error_function_Knn(ifname)

#Create 3 empty lists for each of the models, in order to append later values in order to create a plot.
# Softmax Regression
    accuracy_reg_scores=[]
    microf1_reg_scores=[]
    macrof1_reg_scores=[]
# Random Forest
    acc_RF_scores=[]
    microf1_RF_scores=[]
    macrof1_RF_scores=[]
# Knn non-scaled
    acc_Knn_scores=[]
    micro_f1_Knn_scores=[]
    macro_f1_Knn_scores=[]
# Knn scaled
    acc_Knn_scaled_scores=[]
    micro_f1_Knn_scaled_scores=[]
    macro_f1_Knn_scaled_scores=[]
# LDA
    acc_LDA_scores=[]
    micro_f1_LDA_scores=[]
    macro_f1_LDA_scores=[]

    nvals=[0.5,0.6,0.7,0.8,0.9] #train-test splits
    for n in nvals:
        #all models are run for the particular train-test split
        acc_reg, microf1_reg, macrof1_reg= SoftmaxRegression(ifname, n, 0.001, iterations=3000, regularisation=1)
        acc_rf, microf1_rf, macrof1_rf, run_time_rf = RandomForest(ifname, n, 40, 4)
        acc_Knn, microf1_Knn, macrof1_Knn, run_time_Knn = Knn(ifname,n,25)
        acc_Knn_scaled, microf1_Knn_scaled, macrof1_Knn_scaled, run_time_Knn_scaled = Knn_scaled(ifname,n,77)
        acc_LDA, microf1_LDA, macrof1_LDA, runtime_LDA = LDA(ifname, n)

        #all accuracy and F1 score values for each model are appended to their respective lists

        accuracy_reg_scores.append(acc_reg)
        microf1_reg_scores.append(microf1_reg)
        macrof1_reg_scores.append(macrof1_reg)

        acc_RF_scores.append(acc_rf)
        microf1_RF_scores.append(microf1_rf)
        macrof1_RF_scores.append(macrof1_rf)

        acc_Knn_scores.append(acc_Knn)
        micro_f1_Knn_scores.append(microf1_Knn)
        macro_f1_Knn_scores.append(macrof1_Knn)

        acc_Knn_scaled_scores.append(acc_Knn_scaled)
        microf1_Knn_scaled_scores.append(microf1_Knn_scaled)
        macrof1_Knn_scaled_scores.append(macrof1_Knn_scaled)

        acc_LDA_scores.append(accLDA)
        microf1_LDA_scores.append(microf1_LDA)
        macrof1_LDA_scores.append(macrof1_LDA)

    
    #After the loop has finished the lists are plotted in the accuracy vs train-test split graph
    plt.plot(nvals, accuracy_reg_scores, label='Accuracy on test data (Logistic)')
    plt.plot(nvals, acc_RF_scores, label='Accuracy on test data (Random Forest)')
    plt.plot(nvals,acc_Knn_scores,label='Accuracy on test data (Knn_unscaled)')
    plt.plot(nvals,acc_Knn_scaled_scores, label='Accuracy on test data (Knn_scaled)')
    plt.plot(nvals,acc_LDA_scores, label='Accuracy on test data (LDA)')
    plt.xlabel('Train-test splits')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Train-test Splits for all models')
    plt.legend(loc="center right")
    plt.savefig('plots/Accuracy Graph.png')
    plt.close()

    #Similarly, for F1 Score vs train-test split graph
    plt.plot(nvals, macrof1_LDA_scores, label= 'Macro average f1 score (LDA)', color='black')
    plt.plot(nvals, microf1_LDA_scores, label= 'Micro average f1 score (LDA)', color='purple')
    plt.plot(nvals, macrof1_Knn_scores, label= 'Macro average f1 score (Knn_unscaled)', color='red')
    plt.plot(nvals, microf1_Knn_scores, label= 'Micro average f1 score (Knn_unscaled)', color='tomato')
    plt.plot(nvals, macrof1_RF_scores, label= 'Macro average f1 score (Random Forest)', color='green')
    plt.plot(nvals, microf1_RF_scores, label= 'Micro average f1 score (Random Forest)', color='olive')
    plt.plot(nvals, macrof1_Knn_scaled_scores, label= 'Macro average f1 score (Knn_scaled)', color='brown')
    plt.plot(nvals, microf1_Knn_scaled_scores, label= 'Micro average f1 score (Knn_scaled)', color='peru')
    plt.plot(nvals, microf1_reg_scores, label='Micro average f1 score (Logistic)', color='blue')
    plt.plot(nvals, macrof1_reg_scores, label= 'Macro average f1 score (Logistic)', color='cyan')
    plt.xlabel('Train-test splits')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Train-test splits for all models')
    plt.legend(loc="center right")
    plt.savefig('plots/F1 Score Graph.png')
    plt.close()


    
#For the running of the file through the dataset command
if __name__ == '__main__':
    import sys
    main(sys.argv[1])




        
