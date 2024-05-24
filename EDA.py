#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import statistics as st

#Sampling
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn import svm
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.datasets import make_classification

# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold


# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensembles
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import mannwhitneyu
from scipy.stats import chisquare



warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')



df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.head()


# In[3]:


# Analyze Data
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:',df.columns)
    print('\n')
    print('Data types of each columns: ', df.info())
    

def died(param):
    g = sns.countplot(df[param],hue=df.DEATH_EVENT)
    plt.xticks(rotation=90);
    plt.show()

