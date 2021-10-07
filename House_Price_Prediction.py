#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Data Preprocessing

# ## Importing the libraries and dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Data Sets/HP_train.csv")


# # Data exploration

# In[3]:


df.head()


# In[4]:


df.drop("Id",axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.shape


# ## Dealing with null values

# In[7]:


plt.figure(figsize=(20,5))
sns.heatmap(df.isnull())
plt.show()


# In[8]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[9]:


df.isnull().sum()/df.shape[0]*100


# In[10]:


null_values=df.isnull().sum()/df.shape[0]*100
null_values


# In[11]:


d=null_values[null_values>20].keys()
print(d)


# In[12]:


df=df.drop(columns=d)


# In[13]:


df.head()


# In[14]:


df.columns


# In[15]:


null_values[(null_values>5)&(null_values<18)].keys()


# In[16]:


df['LotFrontage'].unique()


# In[17]:


df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)


# In[18]:


df['LotFrontage'].isnull().sum()


# In[19]:


df['GarageType'].unique()


# In[20]:


df['GarageType'].fillna(df['GarageType'].mode()[0],inplace=True)


# In[21]:


df['LotFrontage'].isnull().sum()


# In[22]:


df['GarageYrBlt'].unique()


# In[23]:


df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(),inplace=True)


# In[24]:


df['GarageYrBlt'].isnull().sum()


# In[25]:


df['GarageFinish'].unique()


# In[26]:


df['GarageFinish'].fillna(df['GarageFinish'].mode()[0],inplace=True)


# In[27]:


df['GarageFinish'].isnull().sum()


# In[28]:


df['GarageQual'].unique()


# In[29]:


df['GarageQual'].fillna(df['GarageQual'].mode()[0],inplace=True)


# In[30]:


df['GarageQual'].isnull().sum()


# In[31]:


df['GarageCond'].unique()


# In[32]:


df['GarageCond'].fillna(df['GarageCond'].mode()[0],inplace=True)


# In[33]:


df['GarageCond'].isnull().sum()


# In[34]:


df.isnull().sum()/df.shape[0]*100


# In[35]:


df.dropna(inplace=True)


# In[36]:


df.isnull().sum()/df.shape[0]*100


# In[ ]:





# ## Handlling Categorical Data

# In[37]:


df_num=df.select_dtypes(['int64','float64'])
df_cat=df.select_dtypes('object')


# In[38]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df_cat:
    df_cat[col]=le.fit_transform(df_cat[col])


# In[39]:


df_cat.head()


# In[40]:


df_num.head()


# In[41]:


new_df=pd.concat([df_num,df_cat],axis=1)


# In[42]:


new_df.head()


# In[43]:


new_df.shape


# In[ ]:





# ## Correlation matrix

# In[44]:


new_df1=new_df.drop('SalePrice',axis=1)


# In[45]:


new_df1.head()


# In[46]:


new_df1.shape


# In[47]:


new_df1.corrwith(new_df['SalePrice']).plot.bar(title="Correlationship with SalePrice",rot=45,grid=True,figsize=(25,5))
plt.show()


# In[48]:


plt.figure(figsize=(15,5))
sns.heatmap(new_df.corr(),annot=True)
plt.show()


# In[ ]:





# ## Splitting the dataset

# In[49]:


x=new_df.drop('SalePrice',axis=1)
y=new_df['SalePrice']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[52]:


x_train


# In[ ]:





# ## Feature scaling

# In[53]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[54]:


x_train


# In[55]:


x_test


# In[ ]:





# # Part 2: Building the model

# In[56]:


def create_model(model):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    #mse=mean_squared_error(y_test,y_pred)
    #print("Mean_Squared_Error: ",mse)
    score=r2_score(y_test,y_pred)
    print("R2_SCORE: ",score)
    return model


# ## Linear Regression

# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lr=LinearRegression()
create_model(lr)


# In[ ]:





# ## RandomForestRegressor

# In[58]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
create_model(rfr)


# In[ ]:





# ## XGBRFRegressor

# In[59]:


from xgboost import XGBRFRegressor
xgb=XGBRFRegressor(reg_alpha=1)
create_model(xgb)


# In[ ]:





# ## Part 3: Hyper parameter tuning

# In[60]:


from sklearn.model_selection import RandomizedSearchCV


# In[61]:


param={
    'n_estimators':[200,400,600,800,1000,1200,1400,1600,1800,2000],
    'max_depth':[10,20,30,40,50,60,70,80,90,100,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['auto','sqrt'],
    'bootstrap':[True,False]
}


# In[62]:


rsc=RandomizedSearchCV(estimator=rfr,param_distributions=param,n_iter=5,n_jobs=-1,cv=5,verbose=3)


# In[63]:


rsc.fit(x_train,y_train)


# In[64]:


rsc.best_estimator_


# ## Part 4: Final model (Random forest regressor)

# In[65]:


rfr1=RandomForestRegressor(max_depth=80, max_features='sqrt', n_estimators=400)
create_model(rfr1)


# In[ ]:




