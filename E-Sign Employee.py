#!/usr/bin/env python
# coding: utf-8

#  ## Part 1: Data preprocessing

# ## Importing the libraries and dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Data Sets/financial_data.csv")


# ## Data exploration

# In[3]:


df.head()


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.drop('entry_id',axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[ ]:





# ## Dealing with the null values

# In[11]:


plt.figure(figsize=(15,5))
sns.heatmap(df.isnull())
plt.show()


# In[ ]:





# ## Encoding the categorical data

# In[12]:


df_num=df.select_dtypes(['int64','float64'])
df_cat=df.select_dtypes('object')


# In[13]:


df_cat.head()


# In[14]:


df_cat['pay_schedule'].unique()


# In[15]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df_cat:
    df_cat[col]=le.fit_transform(df_cat[col])


# In[ ]:





# In[16]:


df_cat.head()


# In[17]:


new_df=pd.concat([df_cat,df_num],axis=1)


# In[18]:


new_df.head()


# In[ ]:





# ## Countplot

# In[19]:


plt.figure(figsize=(15,5))
sns.countplot(data=new_df,x='e_signed')
plt.yticks(new_df['e_signed'].value_counts())
plt.show()


# In[ ]:





# ## Restructure the dataset

# In[20]:


new_df["Month_Employeed"]=new_df['months_employed']+new_df['years_employed']*12


# In[21]:


new_df.head()


# In[22]:


new_df.drop(['months_employed','years_employed'],axis=1,inplace=True)


# In[23]:


new_df.head()


# In[24]:


new_df["Personal_Account_Month"]=new_df['personal_account_m']+new_df['personal_account_y']*12


# In[25]:


new_df.head()


# In[26]:


new_df.drop(['personal_account_m','personal_account_y'],axis=1,inplace=True)


# In[27]:


new_df.head()


# In[ ]:





# ## Correlation matrix and heatmap

# In[28]:


new_df1=new_df.drop('e_signed',axis=1)


# In[29]:


new_df1.head()


# In[30]:


new_df1.corrwith(new_df['e_signed']).plot.bar(title="Correlationship with e_signed",rot=45,grid=True,figsize=(15,5))


# In[31]:


plt.figure(figsize=(15,5))
sns.heatmap(new_df.corr(),annot=True)
plt.show()


# In[ ]:





# ## Splitting the dataset

# In[32]:


x=new_df.drop('e_signed',axis=1)
y=new_df['e_signed']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[ ]:





# ## Feature scaling

# In[35]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[36]:


x_train


# In[37]:


x_test


# In[38]:


pd.Series(y_train).value_counts()


# In[39]:


pd.Series(y_test).value_counts()


# In[40]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()
x_train_sample,y_train_sample=ros.fit_resample(x_train,y_train)
x_test_sample,y_test_sample=ros.fit_resample(x_test,y_test)


# In[41]:


pd.Series(y_train_sample).value_counts()


# In[42]:


pd.Series(y_test_sample).value_counts()


# In[ ]:





# ## Part 2: Building the model

# In[43]:


def create_model(model):
    model.fit(x_train_sample,y_train_sample)
    y_pred=model.predict(x_test_sample)
    print(classification_report(y_test_sample,y_pred))
    print(confusion_matrix(y_test_sample,y_pred))
    return model


# ## 1) Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
lr=LogisticRegression()
create_model(lr)


# ## 2) SVM(Support Vector Machine)

# In[45]:


from sklearn.svm import SVC
svc=SVC(random_state=0)
create_model(svc)


# In[ ]:





# ## 3) Random Forest 

# In[46]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
create_model(rfc)


# In[ ]:





# ## 4) XGBoost

# In[47]:


from xgboost import XGBClassifier
xgb=XGBClassifier(random_state=0,reg_alpha=1)
create_model(xgb)


# In[ ]:





# ## Part 3: Applying Randomized Search to find the best parameters

# In[48]:


from sklearn.model_selection import RandomizedSearchCV


# In[49]:


param={
    'learning_rate':[0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth':[2,3,4,5,6,8,10,12],
    'min_child_weight':[1,3,5,7],
    'gamma':[0.00,0.1,0.2,0.3,0.4],
    'colsample_bytree':[0.3,0.4,0.5,0.6,0.7],
    'n_estimators':[100,200,300,400,500],
    'subsample':[0.5,0.7,1.0]
    
}


# In[50]:


rsc=RandomizedSearchCV(estimator=xgb,param_distributions=param,n_jobs=-1,n_iter=10,cv=5)


# In[51]:


rsc.fit(x_train_sample,y_train_sample)


# In[52]:


rsc.best_estimator_


# In[ ]:





# ## Part 4: Final model (XGBoost Classifier)

# In[54]:


xgb1=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=12,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=300, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=1, reg_lambda=1, scale_pos_weight=1, subsample=1.0,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[55]:


create_model(xgb1)


# In[ ]:




