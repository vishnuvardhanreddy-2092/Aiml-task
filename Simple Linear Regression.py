#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Data Set

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[5]:


data1.info()


# In[6]:


data1.describe()


# In[8]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
sns.boxplot(data=data1["Newspaper"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Newspaper Levels")

sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Newspaper Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()


# In[9]:


plt.scatter(data1["daily"], data1["sunday"])


# In[15]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[16]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[17]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
 
# plotting the regression line
plt.plot(x, y_hat, color = "g")
  
# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




