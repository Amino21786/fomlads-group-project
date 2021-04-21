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
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(df.corr(), annot = True, fmt= '.1g', vmin=-1, vmax=1, cmap='coolwarm', mask=np.triu(df.corr()))
    plt.title('Correlation between Features')
    plt.savefig('plots/CorrelationHeatmap')
    plt.close()




df=pd.read_csv('MobilePricingUpdated.csv') #Importing the preprocessed dataset
#Camera: how many phones vs camera megapixels of primary and front facing camera
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5,color='yellow',label='Front camera')
df['pc'].hist(alpha=0.5,color='blue',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.ylabel('Number of phones')

#plt.show()

