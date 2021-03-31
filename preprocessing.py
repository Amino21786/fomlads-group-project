import numpy as np 
import pandas as pd 
df=pd.read_csv('weatherAUS.csv')
print(df.isnull().mean())
#From the dataset, Sunshine, Evaporation, Cloud9am, Cloud3pm have too many missing values will drop these (all have at least 38% data missing)
df=df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)
#Separate the Categorial and Numerical features to fill in the null balues
df_cat=df[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']]
df_num=df.drop(['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow'], axis = 1)

#for the categorial features, every location has a different windspeed, direction, temperature and pressure
#we can replace the categorial ones with the most frequent values for that location
for col in df_cat.columns.values:
    if df[col].isnull().sum() != 0:
        df_cat[col] = df.groupby(['Location'])[col].apply(lambda x: x.fillna(x.mode().max()))
print(df_cat.isnull().mean())
#some locations have missing values for WindGustDir, so we use the mode of the complete dataset here
df_cat['WindGustDir']=df['WindGustDir'].fillna(df['WindGustDir'].mode().max())
#Replacing Numerical features with mean value based on location same as Categories

for col in df_num.columns.values:
    if df[col].isnull().sum() != 0:
        df_num[col] = df.groupby(['Location'])[col].apply(lambda x: x.fillna(round(x.mean(),1)))


df_num['WindGustSpeed']=df_num['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
df_num['Pressure9am']=df_num['Pressure9am'].fillna(df['Pressure9am'].mean())
df_num['Pressure3pm']=df_num['Pressure3pm'].fillna(df['Pressure3pm'].mean())

#Encoding the Rain columns as OneHot Vectors
d={'Yes':1,'No':0}
df_cat['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df_cat['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

df_updated=pd.merge(df_cat, df_num, left_index=True, right_index=True)
#3 categorical features turn into numerical form
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
df.updated = pd.get_dummies(df_updated, columns=categorical_columns)
df_updated.to_csv('RainAUS.csv', index=False)