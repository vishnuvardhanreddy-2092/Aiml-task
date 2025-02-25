#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[7]:


iris.describe()


# In[8]:


iris.info()


# In[9]:


iris.isna().sum()


# In[10]:


iris.isnull().sum()


# ### Observations
# 

# In[14]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris


# In[ ]:




