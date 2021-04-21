import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def data_analysis(dataset):
    df=pd.read_csv(dataset)
    fig = sns.jointplot(data=df, x='ram', y='price_range', color='blue', kind='kde')
    plt.xlabel('RAM in MB')
    plt.ylabel('Price Range')
    plt.savefig('RAMvsPrice.png')
    return fig
data_analysis('MobilePricingUpdated.csv')


df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
#Camera: how many phones vs camera megapixels of primary and front facing camera
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5,color='yellow',label='Front camera')
df['pc'].hist(alpha=0.5,color='blue',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.ylabel('Number of phones')

plt.show()

