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
#plt.show()