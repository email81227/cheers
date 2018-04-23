import numpy as np
import os
import pandas as pd
import pdb

from DeepModels import *
from keras.utils import np_utils

from os.path import join
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Customized functions
from PublicFunctions import *

train_path = r'D:\DataSet\homburg_audio\train'
test_path = r'D:\DataSet\homburg_audio\test'


train = pd.read_csv(join(train_path, 'train.csv'))
test = pd.read_csv(join(test_path, 'test.csv'))

pdb.set_trace()
X_tr = loader_hd(train_path, r'train_X_256.npy')
y_tr = loader_hd(train_path, r'train_y_256.npy')

X_ts = loader_hd(test_path, r'test_X_256.npy')
y_ts = loader_hd(test_path, r'test_y_256.npy')

lb = LabelEncoder()

y_tr = np_utils.to_categorical(lb.fit_transform(y_tr))

# pdb.set_trace()
# Training
cnn = cnn1D(X_tr, y_tr)

cnn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
cnn.fit(X_tr, y_tr, batch_size=10, epochs=25, validation_split=0.1, verbose=0)

# pdb.set_trace()
prediction = cnn.predict(X_ts)

classes = prediction.argmax(axis=-1)
test['Prediction_Class'] = list(lb.inverse_transform(classes))

print('Accuracy:' + str(sum(test['Category']==test['Prediction_Class'])/len(test)))
pdb.set_trace()