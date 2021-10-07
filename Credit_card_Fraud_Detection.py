#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Data preprocessing

# ## Importing the libraries and the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Data Sets/creditcard.csv")


# In[ ]:





# ## Data Exploration

# In[3]:


df.head()


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


df.head()


# 

# In[6]:


df.shape


# In[ ]:





# ## Dealing with missing values

# In[7]:


plt.figure(figsize=(20,5))
sns.heatmap(df.isnull()) 
plt.show()
df.isnull().sum()/df.shape[0]*100


# In[8]:


df.info()


# In[9]:


df['Class'].unique()


# ## Countplot

# In[10]:


plt.figure(figsize=(20,5))
sns.countplot(data=df,x='Class')
plt.yticks(df['Class'].value_counts())
plt.show()


# In[11]:


new_df=df.drop('Class',axis=1)


# In[12]:


new_df.head()


# In[ ]:





# ## Correlation matrix and heatmap

# In[13]:


new_df.corrwith(df['Class']).plot.bar(title="Correlationship with Class",rot=45,grid=True,figsize=(15,5))
plt.show()


# In[14]:


plt.figure(figsize=(20,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[ ]:





# ## Splitting the dataset

# In[15]:


x=df.drop('Class',axis=True)
y=df['Class']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[ ]:





# ## Feature scaling

# In[18]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[19]:


x_train


# In[20]:


x_test


# In[21]:


pd.Series(y_train).value_counts()


# In[22]:


pd.Series(y_test).value_counts()


# In[23]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()
x_train_sample,y_train_sample=ros.fit_resample(x_train,y_train)
x_test_sample,y_test_sample=ros.fit_resample(x_test,y_test)


# In[24]:


pd.Series(y_train_sample).value_counts()


# In[25]:


pd.Series(y_test_sample).value_counts()


# In[ ]:





# ## Part 2: Building the model

# In[26]:


def create_model(model):
    model.fit(x_train_sample,y_train_sample)
    y_pred=model.predict(x_test_sample)
    print(classification_report(y_test_sample,y_pred))
    print(confusion_matrix(y_test_sample,y_pred))
    return model


# ## 1) Logistic regression

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
lr=LogisticRegression()
create_model(lr)


# ## 2) Random forest

# In[28]:


from sklearn.ensemble import RandomForestClassifier
ros=RandomForestClassifier()
create_model(ros)


# In[ ]:





# ## 3) XGBoost classifier

# In[29]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
create_model(xgb)


# In[ ]:





# ## Part 3: Final model (Logistic Regression)

# In[30]:


from sklearn.model_selection import RandomizedSearchCV


# In[31]:


param={'penalty':['l1', 'l2', 'elasticnet', 'none'],
    'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
    'multi_class':['auto', 'ovr',],
    'max_iter':[100,200,300,400,500,600]}


# In[32]:


rsc=RandomizedSearchCV(estimator=lr,param_distributions=param,n_jobs=-1,n_iter=10,cv=5,verbose=5)


# In[33]:


rsc.fit(x_train_sample,y_train_sample)


# In[34]:


rsc.best_estimator_


# In[35]:


lr1=LogisticRegression(max_iter=200, penalty='l1', solver='saga')
create_model(lr1)


# In[ ]:




