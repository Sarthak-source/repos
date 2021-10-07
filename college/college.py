import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns  #Python library for Vidualization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../Mall_Customers.csv"))

# Any results you write to the current directory are saved as output.

#Import the dataset
#Input from the user

dataset = pd.read_csv(r'college.csv')
Q1 = dataset.quantile(0.05)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1

#Removing the Duplicates
dataset.duplicated().sum()
dataset.drop_duplicates(inplace=True)


#Remove the NaN values from the dataset
dataset.isnull().sum()
dataset.dropna(how='any',inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset[dataset.columns[0]]=le.fit_transform(dataset[dataset.columns[0]])
dataset.rename(columns={'Unnamed: 0':'colleges'})

index_outliers= dataset.loc[((dataset< (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)].index.tolist() #Index of the outliers 
outlier_values=dataset.loc[index_outliers]    #List of values of outliers
remove_outliers=dataset[~((dataset< (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]   # Removing outliers from dataframe and get the dataframe without outliers

dataset=remove_outliers.reset_index()


X1=1
X2=14


#Exploratory Data Analysis
#As this is unsupervised learning so Label (Output Column) is unknown

dataset.head(10) #Printing first 10 rows of the dataset
dataset.shape
dataset.info()
dataset.describe()
dataset.drop('index',axis=1)



X= dataset.iloc[:, [X1,X2]].values


#Building the Model
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod
#to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation
from sklearn.cluster import KMeans
wcss=[]

#we always assume the max number of cluster would be 10
#you can judge the number of clusters by doing averaging
###Static code to get max no of clusters

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters
    
    
#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

kmeansmodel = KMeans(n_clusters= 2, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

fig=plt.figure(figsize=(10,10),edgecolor='k')

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],marker='*',s = 150, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],marker='v',s = 150, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],marker='o', s = 150, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],marker='^', s = 150, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],marker='>', s = 150, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s = 200, c = 'black', label = 'Centroids')
# Set figure width to 12 and height to 9
a=list(dataset.columns)

plt.xlabel(a[X1],fontsize=25)
plt.ylabel(a[X2],fontsize=25)
