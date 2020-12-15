# Python 3.7.4 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Windows 10
# -*- coding: utf-8 -*-
"""
@author: Kanak Choudhury
"""

path = 'Kanak_lab1\\';

import numpy as np
import re
from sklearn.metrics import confusion_matrix

# function to classify new data

def classify(indx, total_class, vocab_len, dt, prob, condi_prob):
    a1 = np.zeros([1,vocab_len]);
    a2 = np.where(dt[:,0]==indx+1)[0];
    a1[0,dt[a2,1]-1] = dt[a2,2];
    res = np.argmax(np.transpose(np.log(prob)) + sum(
            np.transpose(a1*np.log(np.transpose(condi_prob)))))+1;
    return int(res);

# data reading

print(" Reading Data...")
tr = open(path+'train_data.csv', 'r').read().splitlines();
dim_tr = np.shape(tr)[0];

ts = open(path+'test_data.csv', 'r').read().splitlines();
dim_ts = np.shape(ts)[0];

tr_lb = open(path+'train_label.csv', 'r').read().splitlines();
dim_tr_lb = np.shape(tr_lb)[0];

ts_lb = open(path+'test_label.csv', 'r').read().splitlines();
dim_ts_lb = np.shape(ts_lb)[0];

vocab = open(path+'vocabulary.txt', 'r').read().splitlines();
dim_vocab = np.shape(vocab)[0];

maxdim = max(dim_tr, dim_ts, dim_tr_lb, dim_ts_lb)


tr_dt=np.zeros([dim_tr,3]);
ts_dt=np.zeros([dim_ts,3]);
tr_dt_lb=np.zeros([dim_tr_lb]);
ts_dt_lb=np.zeros([dim_ts_lb]);

for i in range(0,dim_tr):
    if i < dim_tr:
        vec_tr = re.split(r'\W+', tr[i])
        tr_dt[i,:]=vec_tr;

    if i < dim_ts:
        vec_ts = re.split(r'\W+', ts[i])
        ts_dt[i,:]=vec_ts;

    if i < dim_tr_lb:
        vec_tr_lb = re.split(r'\W+', tr_lb[i])
        tr_dt_lb[i]=vec_tr_lb[0];

    if i < dim_ts_lb:
        vec_ts_lb = re.split(r'\W+', ts_lb[i])
        ts_dt_lb[i]=vec_ts_lb[0];
tr_dt = tr_dt.astype(int)
ts_dt = ts_dt.astype(int)
tr_dt_lb = tr_dt_lb.astype(int)
ts_dt_lb = ts_dt_lb.astype(int)


del i, tr, vec_tr
del ts, vec_ts
del tr_lb, vec_tr_lb
del ts_lb, vec_ts_lb


# Prior Probability
omega = open(path+'map.csv', 'r').read().splitlines();
dim_omega = np.shape(omega)[0]

nclass_tr=[tr_dt_lb.tolist().count(i+1) for i in range(0,dim_omega)]

priorl_prob = [nclass_tr[i]/dim_tr_lb for i in range(0,dim_omega)];

nclass_ts=[ts_dt_lb.tolist().count(i+1) for i in range(0,dim_omega)]

for i in range(0,dim_omega):
    print(" P(omega = %2d) = %.4f" %(i+1,priorl_prob[i]))

cond_prob = np.zeros([dim_vocab,dim_omega]);
cont_vec = np.zeros([dim_vocab,dim_omega]);


# Conditional Probability
print("Conditional Probablities...")
for j in range(0,dim_omega):
    b1 = np.where(tr_dt_lb == j+1)[0];
    for i in range(0,dim_tr):
        if tr_dt[i,0] in b1:
            cont_vec[tr_dt[i,1]-1,j] = cont_vec[tr_dt[i,1]-1,j] + tr_dt[i,2];
    total_count = sum(cont_vec[:,j]);
    cond_prob[:,j] = (cont_vec[:,j] + 1)/(total_count + dim_vocab);

del b1, total_count

# traing Data Classification
print("\n Classifying Train Data...")
tr_pred = np.zeros(dim_tr_lb);
for k in range(0,dim_tr_lb):
    tr_pred[k] = classify(k, dim_omega, dim_vocab, tr_dt, priorl_prob, cond_prob)


tr_conf_mat = confusion_matrix(tr_dt_lb, tr_pred)
print(" Confusion matrix for train data-")
print(tr_conf_mat)
print("\n Overall accuracy  for train data = %.4f" %(np.trace(tr_conf_mat)/dim_tr_lb))
for i in range(0,dim_omega):
    print(" Class accuracy for group %.2d = %.4f" %(i+1,tr_conf_mat[i,i]/nclass_tr[i]))


# Test Data Classification
print("\n Classifying Test Data...")
ts_pred = np.zeros(dim_ts_lb);
for k in range(0,dim_ts_lb):
    ts_pred[k] = classify(k, dim_omega, dim_vocab, ts_dt, priorl_prob, cond_prob)

ts_conf_mat = confusion_matrix(ts_dt_lb, ts_pred)
print(" Confusion matrix for test data-")
print(ts_conf_mat)
print("\n Overall accuracy for test data = %.4f" %(np.trace(ts_conf_mat)/dim_ts_lb))
for i in range(0,dim_omega):
    print(" Class accuracy for group %.2d = %.4f" %(i+1,ts_conf_mat[i,i]/nclass_ts[i]))