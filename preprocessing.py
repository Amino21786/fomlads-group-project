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
df=df.drop(['three_g'], axis=1)

#Finally update the csv for use
df.to_csv('MobilePricingUpdated.csv', index=False)



"""
Thinking of putting this train and test split idea here so we dont need to keep writing this code for each model
def train_test_split(dataset):
    i=[0.5,0.7,0.75,0.8,0.85,0.9,0.95,0.99] #train test split takes different values, was previously : train_test_split = 0.7
    index_array=np.arange(len(ys))
    train_index_array = index_array[:int(len(index_array) * train_test_split)]
    test_index_array = index_array[int(len(index_array) * train_test_split):]
# get train and test subsets
    x_train = xs[train_index_array]
    y_train = ys[train_index_array]
    x_test = xs[test_index_array]
    y_test = ys[test_index_array]
    start=time.process_time()


could use this, very detailed report version of df.describe
import pandas_profiling as pandas_pf
pandas_pf.ProfileReport(df)

"""