#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[4]:


# Mean value of SAT score
np.mean(df["SAT"])


# In[6]:


# Median value of GradRate score
np.median(df["GradRate"])


# In[7]:


# Standard deviation of data
np.std(df["GradRate"])


# In[8]:


# Find the variance
np.var(df["GradRate"])


# In[9]:


df.describe()


# In[ ]:




