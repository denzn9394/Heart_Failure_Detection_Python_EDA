#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


data


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


#checking for null values
data.isnull().sum()


# In[ ]:


sns.countplot(data['DEATH_EVENT'])


# In[ ]:


#checking the attribute types
data.dtypes


# In[ ]:


# Show skewness and kurtusis of a particular columns
print('-----------Skewness--------------')
print(data.skew())


# In[ ]:


# Investigate all the elements whithin each Feature 
#How many unique values are they , and what are they  ? 

#efficient and scalable way
for column in data:
    unique_values = np.unique(data[column])
    unique_count = len(unique_values)
    if unique_count <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column, unique_count, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, unique_count))

