import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def preprocessing(dataset):
    df=pd.read_csv(dataset)
    print(df.isnull().sum()) #turns out there is no null values
    #We could remove the three_g column as when a phone has 4_g it automatically has 3_g. Also in everyday now, 4_g is more relevant
    df_updated=df.drop(['three_g'], axis=1)
    #check if the dataset is balanced across the 4 classes
    print(df_updated["price_range"].value_counts()) #turns out it is evenly distributed between the 4 classes

    #Finally update the csv for use
    df_updated.to_csv('MobilePricingUpdated.csv', index=False)


def data_analysis(dataset):
    df=pd.read_csv(dataset)
    fig = sns.jointplot(data=df, x='ram', y='price_range', color='blue', kind='kde')
    plt.xlabel('RAM in MB')
    plt.ylabel('Price Range')
    plt.savefig('RAMvsPrice.png')
    return fig
data_analysis('MobilePricingUpdated.csv')

#Camera: how many phones vs camera megapixels of primary and front facing camera
df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5,color='yellow',label='Front camera')
df['pc'].hist(alpha=0.5,color='blue',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.ylabel('Number of phones')

#plt.show()
