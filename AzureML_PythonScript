# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

def azureml_main(dataframe1 = None, dataframe2 = None):
    
    #kmeans = KMeans(n_clusters = 3, random_state = 110)
    
    data = dataframe1
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(data)
    
    centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(6*1.618, 6))

    plt.scatter(data['x'], data['y'], marker='o', s=10)
    
    plt.scatter(centers[:,0], centers[:,1], s=100, c='red')
    
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('Sample (Perfect K-Means)', fontsize= 18)
    plt.savefig('scatter.png')
    
    print(centers)
        
    print('This is test')

   
    return pd.DataFrame(kmeans.cluster_centers_)
