# Speech Emotion Recognition

**Authors:**
Ivan Melnikov, Enam Qassem

![image](https://user-images.githubusercontent.com/56500870/144304551-8f379c94-ffef-4c7b-8dbc-e9670c8df75f.png)


For the task of recognizing emotions from audio files we used the dataset named **RAVDESS**.

**Source**: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio

It consists of files with recorded speech of 24 professional actors (12 female, 12 male), 60 tracks per actor x 24 actors = **1440 in total**.

**Speech emotions includes (8 types)**:
- calm
- neutral
- happy
- sad
- angry
- fear
- surprise
- disgust

Each expression is produced at two levels of **emotional intensity** (normal, strong), except the neutral one.

Every actor has to perform 8 emotions by saying two sentences. 

**The sentences stated are:**
>1. Kids are talking by the door.
>2. Dogs are sitting by the door.

The length of each audio file is about 4 seconds, the first and last second are usually silenced.

**Libraries required:**
```
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
```
You can also optionally swith off the warnings:
```
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
```
**Using the model**

The model is stored in file 'model.pkl'.
In order to use it, you'll need a python library called 'pickle'. See this example:
```
import pickle

pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)
```
