#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[8]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[9]:


Univ.describe()


# In[15]:


# Read all numeric columns into univ1
Univ1 = Univ.iloc[:,1:]
Univ1


# In[16]:


cols = Univ1.columns


# In[17]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[ ]:




