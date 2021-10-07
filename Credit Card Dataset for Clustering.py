#!/usr/bin/env python
# coding: utf-8

# 
# ## Part 1: Data preprocessing

# In[ ]:





# ## Importing the libraries and the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 


# In[2]:


dataset=pd.read_csv("D:\Data Sets/Credit Card Dataset for Clustering.csv")


# In[3]:


dataset.head()


# ## Data exploration

# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# ## Dealing with missing values

# In[7]:


dataset.isnull().sum()/dataset.shape[0]*100


# In[9]:


dataset['CREDIT_LIMIT'].unique()


# In[12]:


dataset['MINIMUM_PAYMENTS'].unique()


# In[14]:


dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(),inplace=True)


# In[15]:


dataset['CREDIT_LIMIT'].isnull().sum()


# In[16]:


dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(),inplace=True)


# In[17]:


dataset['CREDIT_LIMIT'].isnull().sum()


# In[18]:


dataset.info()


# In[19]:


dataset.drop('CUST_ID',axis=1,inplace=True)


# In[20]:


dataset.head()


# In[ ]:





# ## Correlation matrix

# In[ ]:


plt.figure(figsize=(16,9))
ax = sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[ ]:





# ## Feature scaling

# In[21]:


df=dataset


# In[22]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
dataset=ss.fit_transform(dataset)


# In[23]:


dataset


# In[ ]:





# ## Part 2: Elbow method (finding the optimal number of clusters)

# In[24]:


from sklearn.cluster import KMeans


# In[26]:


wcss=[]#weighted cost of sum of square
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,20),wcss,'bx-')
plt.title('The elbo method')
plt.xlabel("Number of Cluster")
plt.ylabel("wcss")    


# In[ ]:





# ## Part 3: Building the model

# In[31]:


kmeans=KMeans(n_clusters=8,init='k-means++',random_state=0)


# In[32]:


y_means=kmeans.fit_predict(dataset)


# In[33]:


print(y_means)


# In[ ]:





# ## Part 4: Getting the output

# In[35]:


y_means.shape


# In[36]:


y_means=y_means.reshape(len(y_means),1)


# In[38]:


y_means.shape


# In[41]:


b=np.concatenate((y_means,df),axis=1)


# In[42]:


df.columns


# In[43]:


df_final=pd.DataFrame(data=b,columns=['Cluster_Number','BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE'])


# In[44]:


df_final.head()


# In[46]:


df_final.to_csv("Segmented Cluster")


# In[ ]:




