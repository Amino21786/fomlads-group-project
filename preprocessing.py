import numpy as np 
import pandas as pd 

df=pd.read_csv('MobilePricing.csv')
print(df.head()) #look at the first 5 entries of the csv
df.info() #All the data points displayed giving what data type it has
#we see they are all numerical features so no need to convert any categorical features to numerical

print(df.describe()) #Breakdown of each data point, inlcuding mean and standard deviation
#With this we see the columns blue, dual_sim, four_g, three_g, wifi are boolean valued inputs (0 for no and 1 for yes)

#check for any null values
print(df.isnull().sum()) #turns out there is no null values

#We could remove the three_g column as when a phone has 4_g it automatically has 3_g. Also in everyday now, 4_g is more relevant
df_updated=df.drop(['three_g'], axis=1)

#check if the dataset is balanced across the 4 classes
print(df_updated["price_range"].value_counts()) #turns out it is evenly distributed between the 4 classes

#Finally update the csv for use
df_updated.to_csv('MobilePricingUpdated.csv', index=False)

np.random.seed(42) #seed for repeatable results
df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
xs=df.drop(['price_range'], axis=1).to_numpy() #all the numerical data that will be use to predict
ys=df['price_range'].to_numpy() #The target variable


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

"""
could use this, very detailed report version of df.describe
import pandas_profiling as pandas_pf
pandas_pf.ProfileReport(df)

"""