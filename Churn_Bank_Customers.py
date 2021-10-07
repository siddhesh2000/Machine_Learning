#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Data preprocessing

# ## Importing the libraries and dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Data Sets/Churn_Modelling(1).csv")


# ## Data exploration

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.drop("RowNumber",axis=1,inplace=True)


# In[6]:


df.head()


# In[7]:


df.describe()


# ## Dealing with missing data

# In[8]:


plt.figure(figsize=(20,5))
sns.heatmap(df.isnull())
plt.show()
df.isnull().sum()


# In[9]:


df.info()


# ## Encode the categorical data

# In[10]:


df_num=df.select_dtypes(['int64','float64'])
df_cat=df.select_dtypes("object")


# In[11]:


df_num.head()


# In[12]:


df_num.drop("CustomerId",axis=1,inplace=True)


# In[13]:


df_num.head()


# In[14]:


df_cat.head()


# In[15]:


df_cat.drop('Surname',axis=1,inplace=True)


# In[16]:


df_cat.head()


# In[17]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df_cat:
    df_cat[col]=le.fit_transform(df_cat[col])


# In[18]:


df_cat.head()


# In[19]:


new_df=pd.concat([df_num,df_cat],axis=1)


# In[20]:


new_df.head()


# ## Univariate analysis

# In[21]:


new_df1=new_df.drop("Exited",axis=1)


# In[22]:


new_df1.head()


# In[23]:


new_df1.corrwith(new_df["Exited"]).plot.bar(title="Correlationship with Exited",rot=45,grid=True,figsize=(20,5))
plt.show()


# In[24]:


plt.figure(figsize=(20,5))
sns.heatmap(new_df.corr(),annot=True)
plt.show()


# ## Splitting the dataset

# In[25]:


x=new_df.drop("Exited",axis=1)
y=new_df["Exited"]


# In[26]:


x.head()


# In[27]:


y.head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[30]:


x_train


# In[31]:


x_test


# In[32]:


y_train


# In[33]:


y_test


# ## Feature scaling

# In[34]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[35]:


x_train


# In[36]:


x_test


# ## PCA (Pricipal Component Analysis)

# In[37]:


from sklearn.decomposition import PCA


# In[38]:


pca=PCA(n_components=None,random_state=0)


# In[39]:


x_train_pca=pca.fit_transform(x_train)
x_test_pca=pca.transform(x_test)


# In[40]:


explained_variance=pca.explained_variance_ratio_
print(explained_variance)


# In[41]:


pca1=PCA(n_components=6,random_state=1)


# In[42]:


x_train_pca1=pca1.fit_transform(x_train)
x_test_pca1=pca1.transform(x_test)


# In[ ]:





# ## Balancing of Output data

# In[43]:


plt.figure(figsize=(20,5))
sns.countplot(data=new_df,x=new_df["Exited"])
plt.yticks(new_df['Exited'].value_counts())
plt.show()


# In[44]:


pd.Series(y_train).value_counts()


# In[45]:


pd.Series(y_test).value_counts()


# In[46]:


from imblearn.over_sampling import RandomOverSampler


# In[47]:


ros=RandomOverSampler()
x_train_sample,y_train_sample=ros.fit_resample(x_train_pca1,y_train)
x_test_sample,y_test_sample=ros.fit_resample(x_test_pca1,y_test)


# In[48]:


pd.Series(y_train_sample).value_counts()


# In[49]:


pd.Series(y_test_sample).value_counts()


# ## Part 2: Building the model

# ## 1) Logistic regression

# In[50]:


def create_model(model):
    model.fit(x_train_sample,y_train_sample)
    y_pred=model.predict(x_test_sample)
    print(classification_report(y_test_sample,y_pred))
    print(confusion_matrix(y_test_sample,y_pred))
    return model
          


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
lr=LogisticRegression(max_iter=200,verbose=5)
create_model(lr)


# ## Cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=lr, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## 2) DecisionTreeClassifier

# In[52]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
create_model(dt)


# ## Cross Vailidation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=dt, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## DecisionTreeClassifier 1

# In[53]:


dt1=DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_leaf=45)
create_model(dt1)


# ## Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=dt1, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## DecisionTree Classifier2

# In[54]:


dt2=DecisionTreeClassifier(max_depth=8,min_samples_leaf=45)
create_model(dt2)


# ## Cross Validation

# In[3]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=dt2, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## XGBClassifier 

# In[55]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
create_model(xgb)


# ## Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xgb, X=x_train, y=y_train, cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# ## Part 3: Randomized Search to find the best parameters (XGBoost Classifier)

# In[56]:


from sklearn.model_selection import RandomizedSearchCV


# In[57]:


param={
    'leraning_rate':[0.05,0.1,0.15,0.20,0.25,0.30],
    'max_depth':[2,3,4,5,6,8,10,12,15],
    'min_child_weight':[1,3,4,5,8],
    'n_estimators':[100,200,300,400,500,600,700,800,900,1000],
    'gamma':[0.0,0.1,0.2,0.3,0.4],
    'colsample_bytree':[0.2,0.3,0.4,0.5,0.6]
      }


# In[58]:


rsc=RandomizedSearchCV(estimator=xgb,param_distributions=param,n_jobs=-1,n_iter=5,scoring='roc_auc')


# In[59]:


rsc.fit(x_train_sample,y_train_sample)


# In[60]:


rsc.best_estimator_


# In[ ]:





# ## Part 4: Final Model (XGBoost Classifier)

# In[62]:


xgb1=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0.3, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, leraning_rate=0.05, max_delta_step=0,
              max_depth=8, min_child_weight=4,
              monotone_constraints='()', n_estimators=700, n_jobs=4,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)


# In[63]:


create_model(xgb1)


# In[ ]:




