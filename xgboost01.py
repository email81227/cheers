# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:46:25 2018

@author: SUSMITA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:13:10 2018

@author: SUSMITA
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


import os
import pandas as pd
import pdb
from pylab import *


from keras.utils import np_utils

from os.path import join
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Customized functions
from PublicFunctions import *

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_path = r'A:\sem2\advanced ML\group project\USC_dataset\USC_dataset'
test_path = r'A:\sem2\advanced ML\group project\USC_dataset\USC_dataset'


train = pd.read_csv(join(train_path, 'train.csv'))
test = pd.read_csv(join(test_path, 'test.csv'))

# pdb.set_trace()

X_ts = loader_hd(test_path, r'mfccs_test_X_all_256.npy')
X_tr = loader_hd(train_path, r'mfccs_train_X_all_256.npy')
y_tr = loader_hd(train_path, r'mfccs_train_y_all_256.npy')
#y_ts = loader_hd(test_path, r'test_y_256.npy')



X_tr=X_tr.mean(axis=1);
X_ts=X_ts.mean(axis=1);
lb = LabelEncoder()
y_tr=lb.fit_transform(y_tr);
#y_ts=lb.fit_transform(y_ts);



dtrain = xgb.DMatrix(X_tr, label=y_tr)
dtest = xgb.DMatrix(X_ts)

#xgboost
param = {
    'max_depth': 20,  # the maximum depth of each tree
    'eta': 0.2,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 9}  # the number of classes that exist in this datset
num_round = 100  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')


preds = bst.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])


classes = best_preds[:len(test)].argmax(axis=-1)
test['Class'] = list(lb.inverse_transform(classes))
test.to_csv(join(r'A:\sem2\advanced ML\group project\USC_dataset\USC_dataset', 'sub_CNN_2D_1.csv'), index=False)
#print(sum(y_ts == best_preds)/len(y_tr))
# Plot outputs
#plot(X_ts, y_ts, color="red", lw=2, linestyle='-')
#plot(X_ts, diabetes_y_pred, 'g*-')

#plt.xticks(())
#plt.yticks(())

plt.show()
