#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


# Mean value of SAT score
np.mean(df["SAT"])


# In[4]:


# Median value of GradRate score
np.median(df["GradRate"])


# In[5]:


# Standard deviation of data
np.std(df["GradRate"])


# In[6]:


# Find the variance
np.var(df["GradRate"])


# In[7]:


df.describe()


# In[8]:


# Visualization
# Visualize the GradRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.hist(df["GradRate"])


# In[10]:


plt.figure(figsize=(90,100))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[15]:


#### Visualization using boxplot
s = [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[18]:


plt.boxplot(scores, vert=False)


# In[19]:


s = [20,15,10,25,30,35,28,40,45,60,120,130]
scores = pd.Series(s)
scores


# In[20]:


plt.boxplot(scores, vert=False)


# In[22]:


df = pd.read_csv("universities.csv")
df


# In[29]:


plt.figure(figsize=(3,2))
plt.boxplot(df["SAT"])


# In[ ]:


plt.figure(fig)

