'''
    Created by Bowen
'''

import os
import pandas as pd
import random
import uuid

from pandas import DataFrame
from sklearn.cross_validation import train_test_split


def split_dataset(dataframe):
    songs = ['alternative', 'blues', 'electronic', 'folkcountry', 'funksoulrnb', 'jazz', 'pop', 'raphiphop', 'rock']
    df_train_new = DataFrame()
    df_test_new = DataFrame()

    for song in songs:
        X = df[df['Category'] == song].iloc[:,:].values
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
        df_train = DataFrame(X_train)
        df_test = DataFrame(X_test)
        df_train_new = df_train_new.append(df_train, ignore_index=True)
        df_test_new = df_test_new.append(df_test, ignore_index=True)
    df_train_new.columns = ['Name', 'Category', 'ID']
    df_test_new.columns = ['Name', 'Category', 'ID']

    return df_train_new, df_test_new


df = pd.read_csv('songs.csv', index_col=0)

train, test = split_dataset(df)