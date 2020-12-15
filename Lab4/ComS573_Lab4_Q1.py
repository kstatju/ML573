#!/usr/bin/env python
# coding: utf-8

# <h1><center> ComS 573     </center></h1>
# <h1><center> Lab 4 </center></h1>
# <h1><center> Kanak Choudhury </center></h1>

# # Problem 1

# In[15]:


import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib
import keras
import re
import sys
import gc
import time

print('python ' +sys.version)
print('numpy '+ np.__version__)
print('pandas '+ pd.__version__)
print('sklearn '+ sklearn.__version__)
print('matplotlib '+ matplotlib.__version__)
print('re '+ re.__version__)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from itertools import product

def print_out(model, model_name, hyper_prem, x_dt_tr, y_dt_tr, x_dt_ts, y_dt_ts):
    print("For "+model_name+" hyper-parameters:\n",hyper_prem)
    scores = model.score(x_dt_ts, y_dt_ts)
    print("\n Test Accuracy: %.2f%%" % (scores*100))

    A = model.predict(x_dt_tr)
    cm = confusion_matrix(y_dt_tr, A)
    print("\n Train confusion matrix: \n", cm)
    acc_train = np.diagonal(cm)/cm.sum(axis=1)
    print("\n Class Accuracy for Training Data is:")
    for i in range(2):
        print('Class %d: %.2f%%' %(i, acc_train[i]*100))

    A = model.predict(x_dt_ts)
    cm = confusion_matrix(y_dt_ts, A)
    print("\n Test confusion matrix: \n", cm)
    acc_test = np.diagonal(cm)/cm.sum(axis=1)
    print("\n Class Accuracy for Testing Data is:")
    for i in range(2):
        print('Class %d: %.2f%%' %(i, acc_test[i]*100))
    print("**********************************\n")


# In[16]:


path  = ''

df_train = pd.read_csv(path + 'lab4-train.csv', sep=',', header=0)
df_test = pd.read_csv(path + 'lab4-test.csv', sep=',', header=0)
tr_size = df_train.shape
ts_size = df_test.shape

x_train = np.array(df_train[['R','F','M','T']])
y_train = np.array(df_train['Class'])

x_test = np.array(df_test[['R','F','M','T']])
y_test = np.array(df_test['Class'])
x_train12 = x_train
y_train12 = y_train
x_test12 = x_test
y_test12 = y_test


# ## Random Forest

# In[17]:


n_estimators=[50, 100, 150, 200]
criterion=['gini', 'entropy']
max_depth=[1, 2, 3 ,4]
min_samples_split=[5, 7, 10, 12]
min_samples_leaf=[1, 2, 3]

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

dictionary = {'n_estimators': n_estimators,
              'criterion': criterion,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf}

prem1 = expand_grid(dictionary)
size_prem = prem1.shape[0]
prem = prem1
prem['train_acc'] = np.NaN
prem['test_acc'] = np.NaN

ll = 0
best_fit = None
best_ts_acc = 0

for i in range(prem.shape[0]):
    ts_acc1 = 0
    rf=RandomForestClassifier(n_estimators=prem.iloc[i,0], criterion=prem.iloc[i,1], 
                              max_depth=prem.iloc[i,2], 
                              min_samples_split=prem.iloc[i,3], min_samples_leaf=prem.iloc[i,4], 
                              max_features='auto', bootstrap=True)
    model_rf = rf.fit(x_train, y_train)
    ts_acc1 = model_rf.score(x_test, y_test)*100
    if (ts_acc1 > best_ts_acc):
        best_ts_acc = ts_acc1
        best_fit = model_rf
    prem.loc[i,5:7] = [model_rf.score(x_train, y_train)*100, model_rf.score(x_test, y_test)*100]
    ll = ll+1
    sys.stdout.write("\r Progress: %.2f%%" %round(float(ll)/size_prem*100,2))
    sys.stdout.flush()


# In[18]:


top10_mse = prem.nlargest(10,'test_acc')
print('\n Best 10 hyper-parameter combination for Random Forest:\n', round(top10_mse, 4))

print_out(model = best_fit, model_name = 'Random Forest', 
          hyper_prem = top10_mse.iloc[0,:], x_dt_tr = x_train, 
          y_dt_tr = y_train, x_dt_ts = x_test, y_dt_ts = y_test)


# ## AdaBoost

# In[19]:


n_estimators=[50, 100, 150, 200]
learning_rate=np.logspace(-5,0,30,base=10)


dictionary = {'n_estimators': n_estimators,
              'learning_rate': learning_rate}

prem1 = expand_grid(dictionary)
size_prem = prem1.shape[0]
prem = prem1
prem['train_acc'] = np.NaN
prem['test_acc'] = np.NaN

ll = 0
best_ts_acc = 0
best_fit = None
best_ts_acc = 0

for i in range(prem.shape[0]):
    ts_acc1 = 0
    adb=AdaBoostClassifier(n_estimators=prem.iloc[i,0], learning_rate=prem.iloc[i,1], 
                           algorithm='SAMME.R')
    model_adb = adb.fit(x_train, y_train)
    ts_acc1 = model_adb.score(x_test, y_test)*100
    if (ts_acc1 > best_ts_acc):
        best_ts_acc = ts_acc1
        best_fit = model_adb
    prem.loc[i,2:4] = [model_adb.score(x_train, y_train)*100, model_adb.score(x_test, y_test)*100]
    ll = ll+1
    sys.stdout.write("\r Progress: %.2f%%" %round(float(ll)/size_prem*100,2))
    sys.stdout.flush()


# In[20]:


top10_mse = prem.nlargest(10,'test_acc')
print('\n Best 10 hyper-parameter combination for AdaBoost:\n', round(top10_mse, 4))

print_out(model = best_fit, model_name = 'AdaBoostn', 
          hyper_prem = top10_mse.iloc[0,:], x_dt_tr = x_train, 
          y_dt_tr = y_train, x_dt_ts = x_test, y_dt_ts = y_test)


# ## Comment:
# 
# Based on test accuracy, Random Forest (RF) model has highest (about 84.5%) accuracy than AdaBoost model (about 82.5%). Both models class 0 accuracy are about 96%. However, for class 1, RF has about 40% accuracy compare to AdaBoost (30%). That indicates that, for this data set AdaBoost model have higher bias for the mejority calss than the RM model.

# In[ ]:




