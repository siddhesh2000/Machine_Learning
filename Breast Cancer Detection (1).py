#!/usr/bin/env python
# coding: utf-8

# # Part 1: Data Preprocessing

#  ## Importing the libraries and dataset

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Data Sets\Data Set/Breast cancer prediction data set.csv")


# In[3]:


df.head()


# ## Data exploration

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# ## Dealing with the missing values

# In[7]:


plt.figure(figsize=(20,8))
sns.heatmap(df.isnull())
plt.show()


# In[8]:


df.columns


# In[9]:


df.drop("Unnamed: 32",axis=1,inplace=True)


# In[10]:


df.isnull().sum()


# ## Dealing with categorical data

# In[11]:


df_num=df.select_dtypes(['int64','float64'])
df_cat=df.select_dtypes('object')


# In[12]:


df_num.head()


# In[13]:


df_cat.head()


# In[14]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[15]:


for col in df_cat:
    df_cat[col]=le.fit_transform(df_cat[col])


# In[16]:


df_cat.head()


# ## Univariate analysis

# In[17]:


new_df=pd.concat([df_num,df_cat],axis=1)


# In[18]:


new_df.head()


# In[19]:


new_df.drop("id",axis=1,inplace=True)


# In[20]:


new_df.head()


# In[21]:


new_df1=new_df.drop("diagnosis",axis=1)


# In[22]:


new_df1


# In[23]:


new_df1.corrwith(new_df['diagnosis']).plot.bar(title="Correlationship with Diagnosis",figsize=(20,5),rot=45,grid=True)


# In[24]:


plt.figure(figsize=(20,8))
sns.heatmap(new_df.corr(),annot=True)
plt.show()


# ## Splitting the dataset train and test set

# In[25]:


x=new_df.drop('diagnosis',axis=1)
y=new_df['diagnosis']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[28]:


x_train


# In[29]:


x_test


# In[30]:


y_train


# In[31]:


y_test


# ## Feature scaling

# In[32]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[33]:


x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[34]:


x_train


# In[35]:


x_test


# ## Balancing of Output

# In[36]:


plt.figure(figsize=(15,5))
sns.countplot(data=new_df,x=new_df['diagnosis'])
plt.yticks(new_df['diagnosis'].value_counts())
plt.show()


# In[37]:


pd.Series(y_train).value_counts()


# In[38]:


pd.Series(y_test).value_counts()


# In[ ]:


get_ipython().system(' pip install imblearn')


# In[40]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()


# In[41]:


x_train_sample,y_train_sample=ros.fit_resample(x_train,y_train)
x_test_sample,y_test_sample=ros.fit_resample(x_test,y_test)


# In[42]:


pd.Series(y_train_sample).value_counts()


# In[43]:


pd.Series(y_test_sample).value_counts()


# ## Part 2: Building the model

# In[44]:


def create_model(model):
    model.fit(x_train_sample,y_train_sample)
    y_pred=model.predict(x_test_sample)
    print(classification_report(y_test_sample,y_pred))
    print(confusion_matrix(y_test_sample,y_pred))
    return model


# ## 1) Logistic regression

# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
lr=LogisticRegression()
create_model(lr)


# ## Cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=lr, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## 2) Random forest

# In[46]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
create_model(rfc)


# ## Cross Vailidation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=rfc, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## 3) DecisionTreeClassifier

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
create_model(dt)


# ## Cross Validation 

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=rfc, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## Part 3: Randomized Search to find the best parameters (RandomForestClassifier)

# In[52]:


from sklearn.model_selection import RandomizedSearchCV


# In[53]:


param={'max_depth':[4,5,6,7,8],
       'min_samples_leaf':[50,60,70,80,90,100],
       'n_estimators':[100,200,300,400,500], 
       'criterion':['gini','entropy']}


# In[54]:


roc=RandomizedSearchCV(estimator=rfc,param_distributions=param,n_iter=10,n_jobs=-1,cv=5)


# In[55]:


roc.fit(x_train_sample,y_train_sample)


# In[56]:


roc.best_estimator_


# In[ ]:





# ## Part 4: Final model (RandomForestClassifier)

# In[57]:


rfc1=RandomForestClassifier(criterion='entropy', max_depth=4, min_samples_leaf=60,
                       n_estimators=500)
create_model(rfc1)


# In[ ]:




