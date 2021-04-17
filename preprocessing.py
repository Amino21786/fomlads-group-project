import numpy as np 
import pandas as pd 


df=pd.read_csv('MobilePricing.csv')
print(df.isnull().sum()) #turns out there is no null values

#We could remove the three_g column as when a phone has 4_g it automatically has 3_g. Also in everyday now, 4_g is more relevant
df_updated=df.drop(['three_g'], axis=1)

#check if the dataset is balanced across the 4 classes
print(df_updated["price_range"].value_counts()) #turns out it is evenly distributed between the 4 classes

#Finally update the csv for use
df_updated.to_csv('MobilePricingUpdated.csv', index=False)





"""
could use this, very detailed report version of df.describe
import pandas_profiling as pandas_pf
pandas_pf.ProfileReport(df)

"""