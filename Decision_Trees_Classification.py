#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


# import some dataset from sklearn
iris = datasets.load_iris(as_frame=True).frame


# In[3]:


iris = pd.read_csv("iris.csv")


# In[4]:


iris


# In[5]:


# Bar plot for categorical column "variety"
import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[6]:


iris.info()


# In[7]:


iris[iris.duplicated(keep=False)]


# **Perform label encoding of target column**

# In[8]:


#Encode the three flower classes as 0,1,2
labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:, -1])
iris.head()


# In[9]:


#check the data types after Label encoding
iris.info()


# **Observation**
# - The target column ('variety') is still object type. It needs to be converted to numeric(int)

# In[10]:


#Convert the target column data type to integer
iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[11]:


#Divide the dataset in to x-columns and y-columns
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[12]:


Y


# In[13]:


#Further splitting of data into training and testing data sets
x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state = 1)
x_train


# In[16]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth = None)
model.fit(x_train,y_train)


# In[15]:


#Plot the decision tree
plt.figure(dpi=1500)
tree.plot_tree(model);


# In[17]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[19]:


fn=['sepal length (cm)','sepal width (cm)','petal lendth (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1500)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[20]:


preds = model.predict(x_test)
preds


# In[21]:


print(classification_report(y_test,preds))


# In[ ]:




