#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[4]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[5]:


print(type(data))
print(data.shape)
print(data.size)
           


# In[6]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data.info()


# In[8]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


data1[data1.duplicated()]


# In[11]:


data1.rename({'Solar.R' : 'Solar'}, axis=1, inplace = True)
data1


# In[12]:


data.info()


# In[13]:


data1.isnull().sum()


# In[14]:


cols = data1.columns
colors = ['black', 'yellow' ]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[15]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[16]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


mean_Solar = data1["Solar"].mean()
print("Mean of Solar: ",mean_Solar)


# In[18]:


data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[19]:


# Find the mode values of categorical column (weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[ ]:


# Impute missing values 
data1['Weather'] = data1['Weather'].fillna(mean_weather)
data1.isnull().sum()


# In[23]:


# Detection of outliers in the columns
# Method1: Using histograms and box plots
# create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
# Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

# Plot thehistogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")

# Adjust layout for better spacing
plt.tight_layout()

# Shoe the plot
plt.show()


# In[26]:


## Observations
 The ozone colum has extreme values beyond 81 as seen from box plot
 The same is confirmed from the below right-skewed histogram


# In[28]:


# Detection of outliers in the columns
# Method1: Using histograms and box plots
# create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})

# Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

# Plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




