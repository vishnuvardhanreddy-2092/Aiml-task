#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.describe()


# In[4]:


# Read all numeric columns into univ1
Univ1 = Univ.iloc[:,1:]
Univ1


# In[5]:


cols = Univ1.columns


# In[6]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[7]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0) 
clusters_new.fit(scaled_Univ_df)


# In[8]:


clusters_new.labels_


# In[9]:


set(clusters_new.labels_)


# In[10]:


Univ['clusterid_new'] = clusters_new.labels_


# In[11]:


Univ


# In[12]:


Univ.sort_values(by = "clusterid_new")


# In[13]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations:
# - Cluster 2 appear to be the top rated universities cluster as the cut off score, Top10, SFRatio parameter mean values are highest
# - Cluster 1 appear to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[14]:


Univ[Univ['clusterid_new']==0]


# In[15]:


wcss = []
for i in range(1, 20):
    Kmeans = KMeans(n_clusters=i,random_state=0)
    Kmeans.fit(scaled_Univ_df)
    wcss.append(Kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ##### from the above graph k=3 or 4 which indicates the elbow joint that is the rate of range of slope dicreases

# In[16]:


from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_Univ_df, clusters_new.labels_, metric='euclidean')
score


# In[ ]:




