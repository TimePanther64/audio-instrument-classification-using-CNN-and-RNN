import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from glob import glob
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

class Config:
  def __init__(self, mode='conv', nfilt=26, nfft=512, nfeat=13, rate=16000):
    self.mode = mode
    self.nfilt = nfilt
    self.nfeat = nfeat
    self.nfft = nfft
    self.rate = rate
    self.step = int(rate/10)

df = pd.read_csv('file.csv')
df.drop(['Unnamed: 0', 'Length'], axis=1, inplace=True)
df.set_index('Filename', inplace=True)

for f in df.index:
  rate, signal = wavfile.read('./clean/' + f)
  df.at[f, 'Length'] = signal.shape[0]/rate

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['Length'].mean()

n_samples = 2*int(df.Length.sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config()

if config.mode == 'conv':
  pass
elif config.mode == 'time':
  pass
