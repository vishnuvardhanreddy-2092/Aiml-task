#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


print(type(data))
print(data.shape)
print(data.size)
           


# In[4]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[5]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data.info()


# In[6]:


data1[data1.duplicated(keep = False)]


# In[7]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[8]:


data1[data1.duplicated()]


# In[9]:


data1.rename({'Solar.R' : 'Solar'}, axis=1, inplace = True)
data1


# In[10]:


data.info()


# In[11]:


data1.isnull().sum()


# In[12]:


cols = data1.columns
colors = ['black', 'yellow' ]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[13]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[14]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[15]:


mean_Solar = data1["Solar"].mean()
print("Mean of Solar: ",mean_Solar)


# In[16]:


data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[17]:


# Find the mode values of categorical column (weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


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


# In[19]:


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


# In[20]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[22]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ### METHOD2
#  USING MU +/-3*SIGMA LIMITS (STANDARD DEVIATION METHOD)

# In[23]:


data1["Ozone"].describe()


# In[30]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# ### Observations
# It is observed that only two outliers are identified using std method.
# In box plot method more no of outliers are identified.
# This is because the assumption of normality is not satisfied in this column.

# ### Quantile-Quantile plot for detection of outliers

# In[35]:


import scipy.stats as stats

# Create Q-Q plot
plt.figure(figsize=(9, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outliers Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=10)


# ### Obsevation from Q-Q plot
# ###### The data does not follow normal distribution as the data are deviating significantly away from the red line.
# ###### The data shows a right-skewed distribution and possible outliers

# #### Other visualisations that could help in the detection of outliers

# In[ ]:


# Create a figure for violin plot
sns.voilinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Vio")

