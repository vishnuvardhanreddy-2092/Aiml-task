#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# ### Observations:
# - There is no null values
# - All columns are object data type and categorical in nature
# - As the columns are categorical, we can adopt one-hot-encoding

# In[7]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[8]:


titanic['Age'].value_counts()


# ### Observation:
# - Maximum travellers are the crew, next comes 3rd class travellers are highest, next comes to 1st class travellers, the last ones are the 2nd class travellers

# In[9]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# ### Apriori Algorithm

# In[10]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[11]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[12]:


rules.sort_values(by='lift', ascending = True)


# In[ ]:





# In[13]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




