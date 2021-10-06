# -*- coding: utf-8 -*-
"""Employee Attrition Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ldx2fgGCtCTl1ct7sAXGuYpS6g-J3yeW

# part 1: data preprocessing

dataset link:https://www.kaggle.com/patelprashant/employee-attrition?select=WA_Fn-UseC_-HR-Employee-Attrition.csv

## Importing the libraries and dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/WA_Fn-UseC_-HR-Employee-Attrition.csv')

"""## Data exploration"""

dataset.head()

dataset.shape

dataset.columns

dataset.info()

# Categorical columns
dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

# numerical columns
dataset.select_dtypes(include='int64').columns

len(dataset.select_dtypes(include='int64').columns)

# statistical summary
dataset.describe()

"""## Restructing the dataset"""

dataset.head()

# drop below columns (makes no sence for target variable prediction)
# EmployeeCount, EmployeeNumber, Over18, StandardHours

dataset['EmployeeCount'].nunique()

dataset['EmployeeCount'].unique()

dataset['Over18'].nunique()

dataset['Over18'].unique()

dataset['StandardHours'].unique()

dataset = dataset.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

dataset.shape

"""## Dealing with the missing values"""

dataset.isnull().values.any()

dataset.isnull().values.sum()

"""## Countplot"""

sns.countplot(dataset['Attrition'])
plt.show()

# Employees left the company
(dataset.Attrition == 'Yes').sum()

# Employees with the company
(dataset.Attrition == 'No').sum()

plt.figure(figsize=(20, 20))

plt.subplot(311)
sns.countplot(x='Department', hue='Attrition', data=dataset)

plt.subplot(312)
sns.countplot(x='JobRole', hue='Attrition', data=dataset)

plt.subplot(313)
sns.countplot(x='JobSatisfaction', hue='Attrition', data=dataset)

"""## Corellation matrix and heatmap"""

corr = dataset.corr()

# correlation matrix
plt.figure(figsize=(16, 9))
ax = sns.heatmap(corr, annot=True, cmap='coolwarm')

"""## Dealing with categorical data"""

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

dataset.shape

# one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

dataset.shape

len(dataset.select_dtypes(include='object').columns)

dataset.head()

dataset.rename(columns={'Attrition_Yes':'Attrition'}, inplace=True)

dataset.head()

"""## Splitting the dataset"""

# matrix of features
x = dataset.drop(columns='Attrition')

# target variable
y = dataset['Attrition']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

"""## Feature scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

"""# Part 2: Building the model

## 1) Logistic regression
"""

from sklearn.linear_model import LogisticRegression
classifer_lr = LogisticRegression(random_state=0)
classifer_lr.fit(x_train, y_train)

y_pred = classifer_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## 2) Random forest"""

from sklearn.ensemble import RandomForestClassifier
classifer_rf = RandomForestClassifier(random_state=0)
classifer_rf.fit(x_train, y_train)

y_pred = classifer_rf.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## 3) Support vector machine"""

from sklearn.svm import SVC
classifer_svc = SVC(random_state=0)
classifer_svc.fit(x_train, y_train)

y_pred = classifer_svc.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""# Part 3: Randomized Search to find the best parameters (Logistic regression)"""

from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'penalty':['l1', 'l2', 'elasticnet', 'none'],
    'C':[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 175, 2.0],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter':[50, 100, 500, 2000, 5000]
}

parameters

random_cv = RandomizedSearchCV(estimator=classifer_lr, param_distributions=parameters, n_iter=10, scoring='roc_auc',
                               n_jobs=-1, cv=5, verbose=3)

random_cv.fit(x_train, y_train)

random_cv.best_estimator_

random_cv.best_params_

random_cv.best_score_

"""# Part 4: Final model (Logistic regression)"""

from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=2000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False)
classifer.fit(x_train, y_train)

y_pred = classifer.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""# Part 5: Predicting a single observation"""

dataset.head()

single_obs = [[41, 1102,	1, 2,	2,	94,	3,	2,	4,	5993,	19479,	8,	11,	3,	1,	0,	8,	0,	1,	6,	4,	0,	5, 
               0,	1,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	1]]

classifer.predict(sc.transform(single_obs))

# leaving the company
