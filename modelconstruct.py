import numpy as np 
import pandas as pd 

def train_test_data(dataset, n):
    df=pd.read_csv(dataset) #Importing the preprocessed dataset
    xs=df.iloc[:,:-1].to_numpy() #all the numerical data that will be use to predict (everything but the last column)
    ys=df.iloc[:,-1].to_numpy() #The target variable (the last column, which in our case will be the price_range)
    #forming the train and test data split, through index arrays
    index_array=np.arange(len(ys))
    train_index_array = index_array[:int(len(index_array) * n)]
    test_index_array = index_array[int(len(index_array) * n):]
    # get train and test subsets
    x_train = xs[train_index_array]
    y_train = ys[train_index_array]
    x_test = xs[test_index_array]
    y_test = ys[test_index_array]

    return x_train, y_train, x_test, y_test

def standardscaler(array):
    for i in range(array.shape[1]):
        column=array[:,i]
        mean=column.mean()
        sd=column.std()
        array[:,i]=(array[:,i]-mean)/sd
    return array