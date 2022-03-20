#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import math
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# In[4]:


data= pd.read_csv ('A1_BC_SEER_data.csv')


# In[5]:


data.head(10)


# In[6]:


print("total number of examples or rows in given dataset are: ",data.shape[0])


# In[7]:


print("total number of features or columns in given dataset are: ",data.shape[1])


# In[8]:


# Descriptive statistics

print('Descriptive statistics to understand complete data are shown below:')
#data.describe()


# In[9]:


data.describe().apply(lambda s: s.apply('{0:.5f}'.format))
 


selected_data= pd.DataFrame()
 


selected_data['SEER registry']=data['SEER registry']
selected_data['Marital status at diagnosis']=data['Marital status at diagnosis']
selected_data['Race/ethnicity']=data['Race/ethnicity']
selected_data['Sex']=data['Sex']
selected_data['Primary Site']=data['Primary Site']
selected_data['Laterality']=data['Laterality']
selected_data['Histology recode - broad groupings']=data['Histology recode - broad groupings']
selected_data['ER Status Recode Breast Cancer (1990+)']=data['ER Status Recode Breast Cancer (1990+)']
selected_data['PR Status Recode Breast Cancer (1990+)']=data['PR Status Recode Breast Cancer (1990+)']
selected_data['Breast - Adjusted AJCC 6th Stage (1988-2015)']=data['Breast - Adjusted AJCC 6th Stage (1988-2015)']
selected_data['surg combine']=data['surg combine']

 
print('Descriptive statistics to understand data with selected 11 fields are shown below:')
 

selected_data.describe().apply(lambda s: s.apply('{0:.5f}'.format))

 

print('plotting heatmap to check for correlation')
  


selected_data['Laterality'].unique()
 

corr_matrix = selected_data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm").set_title('Heat Map of correlation matrix')
plt.show()


 

print('plotting scatter matrix below')


 

# uncomment below line before submissions
pd.plotting.scatter_matrix(selected_data, alpha=0.2,figsize=(30,30),diagonal='kde')

 


selected_data = selected_data.drop('ER Status Recode Breast Cancer (1990+)', 1)

 

selected_data.head()

 


selected_data = pd.concat([selected_data,pd.get_dummies(selected_data['Race/ethnicity'], prefix='Race')],axis=1)
selected_data.drop(['Race/ethnicity'],axis=1, inplace=True)


# In[ ]:


selected_data = pd.concat([selected_data,pd.get_dummies(selected_data['Laterality'], prefix='Laterality')],axis=1)
selected_data.drop(['Laterality'],axis=1, inplace=True)


# In[ ]:


selected_data = pd.concat([selected_data,pd.get_dummies(selected_data['PR Status Recode Breast Cancer (1990+)'], prefix='PR')],axis=1)
selected_data.drop(['PR Status Recode Breast Cancer (1990+)'],axis=1, inplace=True)


# In[ ]:





# In[ ]:


scaler = MinMaxScaler()
scaler.fit(selected_data)
selected_data_scaled = scaler.transform(selected_data)

 

print('Starting K-means')


 

print('computing Within-Cluster-Sum of Squared Errors for different values of k to deterine best k value')

 
# How to choose value of K
# use elbow method and calculate Within-Cluster-Sum of Squared Errors (WSS) for different k

dis = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(selected_data_scaled)
    dis.append(kmeanModel.inertia_)


# In[ ]:


plt.figure(figsize=(15,15))
plt.plot(K, dis, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Within-Cluster-Sum of Squared Errors vs k')
plt.show()

 
# perform k means clustering on selected_data for k=4 wrt elbow plot


# In[ ]:


kmeans = KMeans(n_clusters=4).fit(selected_data_scaled)


# In[ ]:


k_means_clusters = kmeans.predict(selected_data_scaled)

 

selected_data['k_means_clusters'] = k_means_clusters

 

pca = PCA(n_components=2)


# In[ ]:


pca_dataframe = pd.DataFrame(pca.fit_transform(selected_data.drop(["k_means_clusters"], axis=1)))


# In[ ]:


pca_dataframe.columns = ["first", "second"]


# In[ ]:


plot = pd.concat([selected_data,pca_dataframe], axis=1, join='inner')


 


cluster0 = plot[plot["k_means_clusters"] == 0]
cluster1 = plot[plot["k_means_clusters"] == 1]
cluster2 = plot[plot["k_means_clusters"] == 2]
cluster3 = plot[plot["k_means_clusters"] == 3]


# In[ ]:


print('shape of cluster 0 is: ',cluster0.shape)
print('shape of cluster 1 is: ',cluster1.shape)
print('shape of cluster 2 is: ',cluster2.shape)
print('shape of cluster 3 is: ',cluster3.shape)


# In[ ]:


x = cluster0['first']
y = cluster0['second']

plt.scatter(x, y, c='coral')

x = cluster1['first']
y = cluster1['second']

plt.scatter(x, y, c='lightblue')

x = cluster2['first']
y = cluster2['second']

plt.scatter(x, y, c='red')

x = cluster3['first']
y = cluster3['second']

plt.scatter(x, y, c='green')




plt.title('Scatter plot after k-means clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


 

selected_data=selected_data.drop(["k_means_clusters"], axis=1)
 

min_samples=selected_data.shape[1]*2


# In[ ]:



k=min_samples if min_samples>2 else 2
nbrs=NearestNeighbors(n_neighbors=k).fit(selected_data_scaled)
dist,indices=nbrs.kneighbors(selected_data_scaled)

 

# # Plotting K-distance Graph
dist = np.sort(dist, axis=0)
dist = dist[:,1]

plt.figure(figsize=(20,10))
plt.plot(dist)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()
 


km=DBSCAN(eps=0.15,min_samples=min_samples)
selected_data['DB_clusters']=km.fit_predict(selected_data_scaled)
selected_data.head()


# In[ ]:


selected_data['DB_clusters']


# In[ ]:


pca = PCA(n_components=2)
pca_dataframe = pd.DataFrame(pca.fit_transform(selected_data.drop(["DB_clusters"], axis=1)))
pca_dataframe.columns = ["first", "second"]
plot = pd.concat([selected_data,pca_dataframe], axis=1, join='inner')


# In[ ]:


DB_cluster0 = plot[plot["DB_clusters"] == 0]
DB_cluster1 = plot[plot["DB_clusters"] == 1]
DB_cluster2 = plot[plot["DB_clusters"] == 2]
DB_cluster3 = plot[plot["DB_clusters"] == 3]


# In[ ]:


print('shape of DB_scan cluster 0 is: ',DB_cluster0.shape)
print('shape of DB_scan cluster 1 is: ',DB_cluster1.shape)
print('shape of DB_scan cluster 2 is: ',DB_cluster2.shape)
print('shape of DB_scan cluster 3 is: ',DB_cluster3.shape)


# In[ ]:


x = DB_cluster0['first']
y = DB_cluster0['second']

plt.scatter(x, y, c='coral')

x = DB_cluster1['first']
y = DB_cluster1['second']

plt.scatter(x, y, c='lightblue')

x = DB_cluster2['first']
y = DB_cluster2['second']

plt.scatter(x, y, c='red')

x = DB_cluster3['first']
y = DB_cluster3['second']

plt.scatter(x, y, c='green')




plt.title('Scatter plot after DB-Scan clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

