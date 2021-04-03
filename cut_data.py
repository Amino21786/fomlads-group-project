import numpy as np
import csv
import matplotlib.pyplot as plt

import pandas as pd
df=pd.read_csv('weatherAUS.csv')
#grouping by location
grouped = df.groupby(df.Location)
#selecting only one location (city under your discretion, only used Albury because
# i'm lazy- might be worth selecting different cities, perhaps neighbouring each other, so that wind speed
# and direction may play part, but otherwise, wind direction may not be useful)
df=grouped.get_group("Albury")
df=df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)
df_cat=df[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']]
df_num=df.drop(['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow'], axis = 1)

#*stolen from Amin, thanks Amin*
d={'Yes':1,'No':0}
df_cat['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df_cat['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#I had an error here where they refused to let me convert categorical features into numerical, please have a look and let me know

df_updated=pd.merge(df_cat, df_num, left_index=True, right_index=True)
#3 categorical features turn into numerical form
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
df.updated = pd.get_dummies(df_updated, columns=categorical_columns)
#data saved to csv in workspace
df_updated.to_csv('edited_weatherAUS_byLocation.csv', index=False)
