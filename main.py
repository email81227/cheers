'''
    Main process of entire project.
'''

import librosa
import os
import pdb
import pandas as pd

from bs4 import BeautifulSoup
from Data_split import *
from os.path import join
from PreProcessingFunctions import *
from PublicFunctions import *

# Save doc for each set
train_path = r'D:\DataSet\homburg_audio\train'
test_path = r'D:\DataSet\homburg_audio\test'

import_path = r'D:\DataSet\homburg_audio\audios'


# MetaData get
xml_path = r'D:\DataSet\homburg_audio\MetaData'
Metas = []
Classes = []
XMLs = get_files(xml_path)
id = 1
for XML in XMLs:
    doc = BeautifulSoup(open(join(xml_path, XML)).read(), 'lxml')
    bands = doc.find_all('band')

    try:
        for band in bands:
            songs = band.find_all('song')

            # Some xml tags with nothing ...
            if songs:
                for song in songs:
                    # Metas.append(dict(loads(dumps(song)), **{'band': band['@name']}))
                    Classes.append({'ID': id,
                                    'Name': song['path'],
                                    'Category': song['genre']})
                    id += 1

    except:
        print('Something wrong, executive currently stopped.')
        pdb.set_trace()

Classes = pd.DataFrame(Classes)

# Training and test data split.
train, test = split_dataset(Classes)
train.columns = ['Category', 'ID', 'Name']
test.columns = ['Category', 'ID', 'Name']

# Audio convert and save to corresponding path
lose_id = []
for i, row in train.iterrows():
    if os.path.exists(join(join(import_path, row['Category']), row['Name'])):
        mp3towav(join(join(import_path, row['Category']), row['Name']), join(train_path, str(row['ID']) + '.wav'))
    else:
        print(row['Name'] + ' in ' + row['Category'] + ' not found.')
        lose_id.append(i)

train.drop(train.index[lose_id], inplace=True)

lose_id = []
for i, row in test.iterrows():
    if os.path.exists(join(join(import_path, row['Category']), row['Name'])):
        mp3towav(join(join(import_path, row['Category']), row['Name']), join(test_path, str(row['ID']) + '.wav'))
    else:
        print(row['Name'] + ' in ' + row['Category'] + ' not found.')
        lose_id.append(i)

test.drop(test.index[lose_id], inplace=True)

pdb.set_trace()

train.to_csv(join(train_path, 'train.csv'), index=False)
test.to_csv(join(test_path, 'test.csv'), index=False)

# Audios get
train = pd.read_csv(join(train_path, 'train.csv'))
test = pd.read_csv(join(test_path, 'test.csv'))

adu_paths = [train_path, test_path]
num_sample = 256
max_len = int(9.0 * 22050 / num_sample)


def parser(row, path, num_mfcc=num_sample, max_len=max_len):
    # function to load files and extract features
    file_name = join(path, str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_best')

        feature = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc)
        # mfccs_mean = np.mean(mfccs, axis=0)
        # mfccs_min = np.min(mfccs, axis=0)
        # mfccs_max = np.max(mfccs, axis=0)
        # mfccs_median = np.median(mfccs, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return [feature, row.Category, row.Name, feature.shape[1]]


def train_len_adjustment(row, max_len, num_mfcc=num_sample):
    if row.length < max_len:
        feature = np.concatenate((row['feature'], np.zeros((num_mfcc, max_len - row['length']))), axis=1)
        return [feature, row.label, row.length, row.name]
    else:
        return [row.feature, row.label, row.length, row.name]


# Training set
pdb.set_trace()
train['length'] = 0
temp = train.apply(lambda x: parser(x, train_path), axis=1)

temp.columns = ['feature', 'label', 'name', 'length']
max_len = max(temp.length)

temp = temp.apply(lambda x: train_len_adjustment(x, max_len), axis=1)

X = np.rollaxis(np.dstack(temp.feature.tolist()), -1)
y = np.array(temp.label.tolist())

saver_hd(X, train_path, r'train_X_' + str(num_sample) + '.npy')
saver_hd(y, train_path, r'train_y_' + str(num_sample) + '.npy')

# Test set
test['length'] = 0
temp = test.apply(lambda x: parser(x, test_path), axis=1)

temp.columns = ['feature', 'label', 'name', 'length']
max_len = max(temp.length)

temp = temp.apply(lambda x: train_len_adjustment(x, max_len), axis=1)

X = np.rollaxis(np.dstack(temp.feature.tolist()), -1)
y = np.array(temp.label.tolist())

saver_hd(X, test_path, r'test_X_' + str(num_sample) + '.npy')
saver_hd(y, test_path, r'test_y_' + str(num_sample) + '.npy')


'''
# set up
for t, adu_path in enumerate(adu_paths):
    feature_list = []
    audios = get_files(adu_path, '.wav')
    for audio in audios:
        # sample_rate, signal = scipy.io.wavfile.read(join(adu_path, audio))  # read file
        # # Keep all time range :10 seconds
        # # signal = signal[0:int(10 * sample_rate)]
        #
        # signal = pre_emphasis(signal)
        # frame, f_len = framing(signal, sample_rate)
        # win = window(frame, f_len)
        # f_bank = fft_filterbank(win, sample_rate)
        # feature = mfccs(f_bank)
        #
        # # Append mfcc into list
        X, sample_rate = librosa.load(join(adu_path, audio), res_type='kaiser_best')

        feature = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_sample)
        if feature.shape[1] > 430:
            feature = feature[:, :430]

        feature_list.append(feature)

    X = np.rollaxis(np.dstack(feature_list), -1)
    y = np.array(train.Category.tolist())

    if t == 0:
        saver_hd(X, train_path, r'train_X_' + str(num_sample) + '.npy')
        saver_hd(y, train_path, r'train_y_' + str(num_sample) + '.npy')
    else:
        saver_hd(X, test_path, r'test_X_' + str(num_sample) + '.npy')
        saver_hd(y, test_path, r'test_y_' + str(num_sample) + '.npy')
'''
