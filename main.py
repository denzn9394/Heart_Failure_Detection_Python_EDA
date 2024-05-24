#!/usr/bin/env python
# coding: utf-8

# In[1]:


from EDA import *


# In[2]:


data=pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


explore_data(data)


# In[4]:


data.isna().sum()


# In[6]:


data.agg(
        {
            "age": ["std","min", "max", "median", "skew", "var","skew"],
            "anaemia": ["std","min", "max", "median", "skew", "var","skew"],
            "creatinine_phosphokinase": ["std","min", "max", "median", "skew", "var","skew"],
            "diabetes": ["std","min", "max", "median", "skew", "var","skew"],
            "ejection_fraction": ["std","min", "max", "median", "skew", "var","skew"],
            "high_blood_pressure": ["std","min", "max", "median", "skew", "var","skew"],
            "platelets": ["std","min", "max", "median", "skew", "var","skew"],
            "serum_creatinine": ["std","min", "max", "median", "skew", "var","skew"],
            "serum_sodium": ["std","min", "max", "median", "skew", "var","skew"],
            "sex": ["std","min", "max", "median", "skew", "var","skew"],
            "smoking": ["std","min", "max", "median", "skew", "var","skew"],
            "time": ["std","min", "max", "median", "skew", "var","skew"],
            "DEATH_EVENT": ["std","min", "max", "median", "skew", "var","skew"]
        }
)


# In[7]:


for column in data:
    unique_values = np.unique(data[column])
    unique_count = len(unique_values)
    if unique_count <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column, unique_count, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, unique_count))


# In[29]:


try:
    col = list(data.columns)
    cat_fea = []
    num_fea = []
    data['age'] = data['age'].astype(int)
    data['platelets'] = data['platelets'].astype(int)
    data1 = data.copy()
    for i in col:
        if len(data[i].unique()) > 6:
            num_fea.append(i)
        else:
            cat_fea.append(i)
    fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))
    for i in range(len(cat_fea)):
        plt.subplot(3,2,i+1)
        sns.distplot(data1[cat_fea[i]],kde_kws = {'bw' : 1});
        title = 'Distribution : ' + cat_fea[i]
        plt.title(title)
    fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))
    for i in range(len(num_fea) - 1):
        plt.subplot(3,2,i+1)
        sns.distplot(data[num_fea[i]])
        title = 'Distribution : ' + num_fea[i]
        plt.title(title)
    plt.show()

    plt.figure(figsize = (4.75,4.55))
    sns.distplot(data1[num_fea[len(num_fea) - 1]],kde_kws = {'bw' : 1})
    title = 'Distribution : ' + num_fea[len(num_fea) - 1]
    plt.title(title);
except ValueError as ve:
    print(ve)


# In[30]:


sns.barplot(x='anaemia',y='DEATH_EVENT',data=data)
plt.title('Relationship between anaemic patient and death_event',weight='bold')


# In[31]:


sns.barplot(x='diabetes',y='DEATH_EVENT',data=data,palette='Greens')
plt.title('Relationship between diabetic patient and death_event',weight='bold')


# In[32]:


sns.barplot(x='high_blood_pressure',y='DEATH_EVENT',data=data,palette='Blues')
plt.title('Relationship between high_blood_pressure and death_event',weight='bold')


# In[33]:


sns.barplot(x='smoking',y='DEATH_EVENT',data=data,palette='Set1')
plt.title('Relationship between smoking and death_event',weight='bold')


# In[34]:


#Correlation
corr = data.corr().sort_values(ascending=False, by="DEATH_EVENT")#, key=lambda x: abs(x), by="DEATH_EVENT")

print("Correlation of all features:")
print (corr["DEATH_EVENT"])

plt.subplots(figsize=(11, 11)) 
sns.heatmap(data.corr(),annot=True)


# In[38]:


#Selecting dependent and independent variables
X=data.iloc[:,:-1]
y=data.iloc[:,-1]


#Performing train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)


# In[39]:


sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


# In[40]:


model=LogisticRegression()
model.fit(X_train,y_train)
#prediction of model
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[41]:


model=SVC()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[42]:


model=RandomForestClassifier(n_estimators=150)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[43]:


model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[44]:


model = MLPClassifier()
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[45]:


model = GaussianNB()
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[46]:


#checking whether the dataset is imbalanced or not
sns.countplot(data['DEATH_EVENT'])


# In[47]:


X = df.drop(["DEATH_EVENT"], axis =1)
y = df["DEATH_EVENT"]
y.value_counts()


# In[48]:


fig1,axs1 = plt.subplots()
labels = ['Survived','Dead']
axs1.pie(y.value_counts(), autopct = '%.2f', labels = labels)


# In[49]:


##                                                  https://www.youtube.com/watch?v=4SivdTLIwHc

ranus = RandomUnderSampler(sampling_strategy = 1)
X_rsmp , y_rsmp = ranus.fit_resample(X,y)

ax=y_rsmp.value_counts().plot.pie(autopct = '%.2f')
t = ax.set_title("Under Sampling Technique")


# In[50]:


y_rsmp.value_counts()


# In[51]:


ranos = RandomOverSampler(sampling_strategy = 1)
X_rsmp , y_rsmp = ranos.fit_resample(X,y)

ax=y_rsmp.value_counts().plot.pie(autopct = '%.2f')
t = ax.set_title("Over Sampling Technique")


# In[ ]:


y_rsmp.value_counts()


# In[56]:


X_train,X_test,y_train,y_test=train_test_split(X_rsmp,y_rsmp,test_size=0.25,random_state=101)


# In[57]:


model=LogisticRegression()
model.fit(X_train,y_train)
#prediction of model
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[58]:


model=SVC()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[59]:


model=RandomForestClassifier(n_estimators=150)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[60]:


model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[61]:


model = MLPClassifier()
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[62]:


model = GaussianNB()
model.fit(X_train, y_train)
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y_test,predictions)*100)


# In[63]:


# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
     print('Feature: %0d, Score: %.05f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,7))
plt.bar([x for x in range(len(importance))], importance, width = 1, edgecolor='black', linewidth=1.2)
plt.show()


# In[64]:


# Call the function
agemw = mannwhitneyu(data['age'], data['DEATH_EVENT'])
anemw = mannwhitneyu(data['anaemia'], data['DEATH_EVENT'])
cremw = mannwhitneyu(data['creatinine_phosphokinase'], data['DEATH_EVENT'])
diamw = mannwhitneyu(data['diabetes'], data['DEATH_EVENT'])
ejemw = mannwhitneyu(data['ejection_fraction'], data['DEATH_EVENT'])
hbpmw = mannwhitneyu(data['high_blood_pressure'], data['DEATH_EVENT'])
plamw = mannwhitneyu(data['platelets'], data['DEATH_EVENT'])
sermw = mannwhitneyu(data['serum_creatinine'], data['DEATH_EVENT'])
sodmw = mannwhitneyu(data['serum_sodium'], data['DEATH_EVENT'])
sexmw = mannwhitneyu(data['sex'], data['DEATH_EVENT'])
smomw = mannwhitneyu(data['smoking'], data['DEATH_EVENT'])
timmw = mannwhitneyu(data['time'], data['DEATH_EVENT'])
print("The Statistics and P-value of Age against Death Event is :", agemw)
print("The Statistics and P-value of anaemia against Death Event is :", anemw)
print("The Statistics and P-value of creatinine_phosphokinase against Death Event is :", cremw)
print("The Statistics and P-value of diabetes against Death Event is :", diamw)
print("The Statistics and P-value of ejection_fraction against Death Event is :", ejemw)
print("The Statistics and P-value of high_blood_pressure against Death Event is :", hbpmw)
print("The Statistics and P-value of platelets against Death Event is :", plamw)
print("The Statistics and P-value of serum_creatinine against Death Event is :", sermw)
print("The Statistics and P-value of serum_sodium against Death Event is :", sodmw)
print("The Statistics and P-value of sex against Death Event is :", sexmw)
print("The Statistics and P-value of smoking against Death Event is :", smomw)
print("The Statistics and P-value of time against Death Event is :", timmw)


# In[65]:


chisquare(data)


# In[66]:


datadf = data.copy()
#datadf.head()
testdf = datadf.drop(['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium',
            'time'], axis=1)


# In[67]:


#Selecting dependent and independent variables
X2=testdf.iloc[:,:-1]
y2=testdf.iloc[:,-1]
print(X2)

#Performing train test split

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.25,random_state=101)


# In[68]:


sc=StandardScaler()
sc.fit(X2_train)
X2_train=sc.transform(X2_train)
X2_test=sc.transform(X2_test)


# In[69]:


model=LogisticRegression()
model.fit(X2_train,y2_train)
#prediction of model
predictions=model.predict(X2_test)
print(classification_report(y2_test,predictions))
print(confusion_matrix(y2_test,predictions))
sns.heatmap(confusion_matrix(y2_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y2_test,predictions)*100)


# In[70]:


model=SVC()
model.fit(X2_train,y2_train)
predictions=model.predict(X2_test)
print(classification_report(y2_test,predictions))
print(confusion_matrix(y2_test,predictions))
sns.heatmap(confusion_matrix(y2_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y2_test,predictions)*100)


# In[71]:


model=RandomForestClassifier(n_estimators=150)
model.fit(X2_train,y2_train)
predictions=model.predict(X2_test)
print(classification_report(y2_test,predictions))
print(confusion_matrix(y2_test,predictions))
sns.heatmap(confusion_matrix(y2_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y2_test,predictions)*100)


# In[72]:


model = KNeighborsClassifier(n_neighbors=4)
model.fit(X2_train, y2_train)
predictions= model.predict(X2_test)
print(classification_report(y2_test,predictions))
print(confusion_matrix(y2_test,predictions))
sns.heatmap(confusion_matrix(y2_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y2_test,predictions)*100)


# In[73]:


model = MLPClassifier()
model.fit(X2_train, y2_train)
predictions= model.predict(X2_test)
print(classification_report(y2_test,predictions))
print(confusion_matrix(y2_test,predictions))
sns.heatmap(confusion_matrix(y2_test,predictions), cbar=False, annot=True)
print("Accuracy:",metrics.accuracy_score(y2_test,predictions)*100)

