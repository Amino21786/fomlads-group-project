import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
def preprocessing(dataset):
    df=pd.read_csv(dataset)
    #We could remove the three_g column as when a phone has 4_g it automatically has 3_g. Also in everyday now, 4_g is more relevant
    df_updated=df.drop(['three_g'], axis=1)
    df_updated.to_csv('MobilePricingUpdated.csv', index=False)


def corrleation_heatmap(dataset):
    df=pd.read_csv(dataset)
    plt.subplots(figsize=(15,13))
    sns.heatmap(df.corr(), annot = True, fmt='.1g', linewidths=.5, vmin=-1, vmax=1, cmap='coolwarm',  cbar_kws= {'orientation': 'horizontal'} )
    plt.title('Correlation between Features')
    plt.savefig('plots/CorrelationHeatmap')
    plt.close()






