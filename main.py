import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from SoftmaxRegression import SoftmaxRegression
from RandomForest import RandomForest
from Knn import Knn
from Knn import Knn_scaled
from LDA import LDA
from plots.plots import rf_hyperparameters, oob_error_rf, Knn_hyperparameters, error_function_Knn
from Knn import ideal_k




def main(ifname):
    plt.style.use('seaborn-whitegrid')

    #Random Forest graphs:
    rf_hyperparameters('MobilePricingUpdated.csv')
    oob_error_rf('MobilePricingUpdated.csv')
    #Knn graphs and prints ideal k:
    ideal_k('MobilePricingUpdated.csv')
    Knn_hyperparameters('MobilePricingUpdated.csv')
    error_function_Knn('MobilePricingUpdated.csv')

#Create 3 empty lists for each of the models, in order to append later values in order to create a plot.

    accuracy_train=[]
    accuracy_test=[]
    microf1_scores=[]
    macrof1_scores=[]

    acc_RF=[]
    microf1_RF=[]
    macrof1_RF=[]
# For the non-scaled data for Knn:
    acc_Knnl=[]
    micro_f1_Knnl=[]
    macro_f1_Knnl=[]
# For the scaled data for Knn:
    acc_Knnl_scaled=[]
    micro_f1_Knnl_scaled=[]
    macro_f1_Knnl_scaled=[]

    acc_LDA =[]
    micro_f1_LDA=[]
    macro_f1_LDA=[]

    nvals=[0.5,0.6,0.7,0.8,0.9]
    for n in nvals:
        acc_train, acc_test, microf1, macrof1= SoftmaxRegression(ifname, n, 0.001, iterations=3000, regularisation=1)
        acc_rf, micro_f1_rf, macro_f1_rf, run_time_rf = RandomForest(ifname, n, 40, 4)
        acc_Knn, micro_f1_Knn, macro_f1_Knn, run_time_Knn = Knn(ifname,n,25)
        acc_Knn_scaled, micro_f1_Knn_scaled, macro_f1_Knn_scaled, run_time_Knn_scaled = Knn_scaled(ifname,n,77)
        accLDA, microf1_LDA, macrof1_LDA, runtime_LDA = LDA(ifname, n)

        accuracy_train.append(acc_train)
        accuracy_test.append(acc_test)
        microf1_scores.append(microf1)
        macrof1_scores.append(macrof1)

        acc_RF.append(acc_rf)
        microf1_RF.append(micro_f1_rf)
        macrof1_RF.append(macro_f1_rf)

        acc_Knnl.append(acc_Knn)
        micro_f1_Knnl.append(micro_f1_Knn)
        macro_f1_Knnl.append(macro_f1_Knn)

        acc_Knnl_scaled.append(acc_Knn_scaled)
        micro_f1_Knnl_scaled.append(micro_f1_Knn_scaled)
        macro_f1_Knnl_scaled.append(macro_f1_Knn_scaled)

        acc_LDA.append(accLDA)
        micro_f1_LDA.append(microf1_LDA)
        macro_f1_LDA.append(macrof1_LDA)

    
    
    

    
    plt.plot(nvals, accuracy_train, label='Accuracy on train data (Logistic)')
    plt.plot(nvals, accuracy_test, label='Accuracy on test data (Logistic)')
    plt.plot(nvals, acc_RF, label='Accuracy on test data (Random Forest)')
    plt.plot(nvals,acc_Knnl,label='Accuracy on test data (Knn)')
    plt.plot(nvals,acc_Knnl_scaled, label='Accuracy on test data (Knn_scaled)')
    plt.plot(nvals,acc_LDA, label='Accuracy on test data (LDA)')
    plt.xlabel('Train-test splits')
    plt.title('Accuracy vs Train-test Splits for all models')
    plt.legend(loc="center right")
    plt.savefig('plots/Accuracy Graph.png')
    plt.close()

   
    plt.plot(nvals, microf1_scores, label='Micro average f1 score (Logistic)', color='red')
    plt.plot(nvals, macrof1_scores, label= 'Macro average f1 score (Logistic)', color='orange')
    plt.plot(nvals, macrof1_RF, label= 'Macro average f1 score (Random Forest)', color='green')
    plt.plot(nvals, microf1_RF, label= 'Micro average f1 score (Random Forest)', color='olive')
    plt.plot(nvals, macro_f1_Knnl, label= 'Macro average f1 score (Knn)', color='blue')
    plt.plot(nvals, micro_f1_Knnl, label= 'Micro average f1 score (Knn)', color='grey')
    plt.plot(nvals, macro_f1_Knnl_scaled, label= 'Macro average f1 score (Knn_scaled)')
    plt.plot(nvals, micro_f1_Knnl_scaled, label= 'Micro average f1 score (Knn_scaled)')
    plt.plot(nvals, macro_f1_LDA, label= 'Macro average f1 score (LDA)', color='pink')
    plt.plot(nvals, micro_f1_LDA, label= 'Micro average f1 score (LDA)', color='purple')
    plt.xlabel('Train-test splits')
    plt.title('F1 Score vs Train-test splits for all models')
    plt.legend(loc="center right")
    plt.savefig('plots/F1 Score Graph.png')
    plt.close()


    
if __name__ == '__main__':
    import sys
    main(sys.argv[1])




        
