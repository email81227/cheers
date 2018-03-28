import tensorflow as tf

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

'''
    Reference:
        Keras installation
            https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-2
            Up to step 4.
        TensorFlow installation
            https://www.tensorflow.org/install/
    Dependencies:
        Python 3.5
        SciPy/ NumPy
        Matplotlib
        TensorFlow (Require CUDA/cuDNN and a listed graphic card if chosen the version GPU support)
'''

# from keras.datasets import mnist
# # Load pre-shuffled MNIST data into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# print(X_train.shape)
# plt.imshow(X_train[0])
# plt.show()

'''
    Reference:
        https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
'''
import glob
import IPython.display as ipd
import librosa
import librosa.display as display
import numpy as np
import os
import pandas as pd
import pdb
import random

from matplotlib import pyplot as plt
from os.path import join
from sklearn.preprocessing import LabelEncoder


doc_path = r'D:\DataSet\UrbanSoundChallenge\train'
train = pd.read_csv(join(doc_path, 'train.csv'))

data_path = r'D:\DataSet\UrbanSoundChallenge\train\Train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test\Test'
sub_path = r'D:\DataSet\UrbanSoundChallenge\submission'

# data, sampling_rate = librosa.load(join(data_path, '2022.wav'))
# plt.figure(figsize=(12, 4))
# # librosa.display doesn't work, instead of that import librosa.display using directly.
# display.waveplot(data, sr=sampling_rate)
# plt.show()
#
# # Random pick
# i = random.choice(train.index)
#
# audio_name = train.ID[i]
# path = join(data_path, str(audio_name) + '.wav')
#
# print('Class: ', train.Class[i])
# x, sr = librosa.load(join(data_path, str(train.ID[i]) + '.wav'))
#
# plt.figure(figsize=(12, 4))
# display.waveplot(x, sr=sr)
# plt.show()
#
# classes_state = train.Class.value_counts() / sum(train.Class.value_counts())
#
# # First submission
test = pd.read_csv(join(r'D:\DataSet\UrbanSoundChallenge\test', 'test.csv'))
# test['Class'] = 'jackhammer'
# test.to_csv(join(sub_path, 'sub01.csv'), index=False)


def parser(row):
    # function to load files and extract features
    file_name = os.path.join(join(data_path, str(row.ID) + '.wav'))

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    feature = mfccs
    label = row.Class

    return [feature, label]

# Step 1 and  2 combined: Load audio files and extract features


# Random pick
# i = random.sample(list(train.index), 300)
# sub_train = train.loc[i]
temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']

# Step 3: Convert the data to pass it in our deep learning model
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))


# Step 4: Run a deep learning model and get results
num_labels = y.shape[1]
filter_size = 3

# build model
# The Sequential model is a linear stack of layers.
model = Sequential()

# The model needs to know what input shape it should expect.
# For this reason, the first layer in a Sequential model (and only the first,
# because following layers can do automatic shape inference) needs to receive
# information about its input shape.
model.add(Dense(32, input_shape=(40,)))  # first layer
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X, y, batch_size=8, epochs=12, validation_data=(X, y))

pdb.set_trace()