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


# In[2]:


data=pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


data


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


#checking for null values
data.isnull().sum()


# In[ ]:


#checking whether the dataset is imbalanced or not
sns.countplot(data['DEATH_EVENT'])


# In[4]:


#checking the attribute types
data.describe()


# # Exploratory Data Analysis

# In[ ]:


#pie-chart of anaemic patients

data.anaemia.value_counts(normalize=True)

prop = data['anaemia'].value_counts()
labels = ['not_anaemic','anaemic']
plt.title('Proportion of anaemic and non-anaemic patients',weight = 'bold')
plt.pie(prop,labels=labels,shadow = True, autopct = '%1.5f%%',wedgeprops= {'edgecolor':'black'})
plt.show()


# In[ ]:


#pie-chart of diabetic patient

data.diabetes.value_counts(normalize=True)

prop = data['diabetes'].value_counts()
labels = ['not_diabeic','diabetic']
plt.title('Proportion of diabetic and non-diabetic patients',weight = 'bold')
plt.pie(prop,labels=labels,shadow = True, autopct = '%1.5f%%',wedgeprops= {'edgecolor':'black'},colors=['red','blue'])
plt.show()


# In[ ]:


#pie chart of people with high_blood_pressure

data.high_blood_pressure.value_counts(normalize=True)

prop = data['high_blood_pressure'].value_counts()
labels = ['normal_bp','high_bp']
plt.title('Proportion of high_bp and low_bp patients',weight = 'bold')
plt.pie(prop,labels=labels,shadow = True, autopct = '%1.5f%%',wedgeprops= {'edgecolor':'black'},colors=['blue','pink'])
plt.show()


# In[ ]:


data.smoking.value_counts(normalize=True)


# In[ ]:


#pie-chart of number of patients

data.sex.value_counts(normalize=True)

prop = data['sex'].value_counts()
labels = ['male','female']
plt.title('Proportion of male and female',weight = 'bold')
plt.pie(prop,labels=labels,shadow = True, autopct = '%1.5f%%',wedgeprops= {'edgecolor':'black'},colors=['blue','orange'])
plt.show()


# In[ ]:


#pie_chart of smokers

data.smoking.value_counts(normalize=True)

prop = data['smoking'].value_counts()
labels = ['non-smoker','smoker']
plt.title('Proportion of smoker and non-smoker',weight = 'bold')
plt.pie(prop,labels=labels,shadow = True, autopct = '%1.5f%%',wedgeprops= {'edgecolor':'black'},colors=['yellow','red'])
plt.show()


# In[ ]:


sns.barplot(x='anaemia',y='DEATH_EVENT',data=data)
plt.title('Relationship between anaemic patient and death_event',weight='bold')


# In[ ]:


sns.barplot(x='diabetes',y='DEATH_EVENT',data=data,palette='Greens')
plt.title('Relationship between diabetic patient and death_event',weight='bold')


# In[ ]:


sns.barplot(x='high_blood_pressure',y='DEATH_EVENT',data=data,palette='Blues')
plt.title('Relationship between high_blood_pressure and death_event',weight='bold')


# In[ ]:


sns.barplot(x='smoking',y='DEATH_EVENT',data=data,palette='Set1')
plt.title('Relationship between smoking and death_event',weight='bold')


# In[ ]:


#Correlation

plt.subplots(figsize=(11, 11)) 
sns.heatmap(data.corr(),annot=True)


# # Feature engineering
# 

# In[ ]:


#Selecting dependent and independent variables
X=data.iloc[:,:-1]
y=data.iloc[:,-1]


#Performing train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)


# In[ ]:


#Scaling the variables

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


# # Model Building

# In[ ]:


#Logistic Regression

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#prediction of model
predictions=model.predict(X_test)


# In[ ]:


#Confusion matrix and classification report

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[ ]:


#building Support Vector Machine

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[ ]:


#building random forest model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=150)
model.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[ ]:


#building  DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[ ]:


#Training the kNN on training set with 4 neighbors

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
predictions= model.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[ ]:




