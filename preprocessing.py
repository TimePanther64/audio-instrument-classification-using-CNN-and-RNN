import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from glob import glob

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)

    i = 0
    for x in range(2):
      for y in range(5):
        axes[x,y].set_title(list(signals.keys())[i])
        axes[x,y].plot(list(signals.values())[i])
        axes[x,y].get_xaxis().set_visible(False)
        axes[x,y].get_yaxis().set_visible(False)
        i+=1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transform', size=16)

    i = 0
    for x in range(2):
      for y in range(5):
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        axes[x,y].set_title(list(fft.keys())[i])
        axes[x,y].plot(freq, Y)
        axes[x,y].get_xaxis().set_visible(False)
        axes[x,y].get_yaxis().set_visible(False)
        i+=1

def plot_fbanks(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)

    i = 0
    for x in range(2):
      for y in range(5):
        axes[x,y].set_title(list(fbank.keys())[i])
        axes[x,y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
        axes[x,y].get_xaxis().set_visible(False)
        axes[x,y].get_yaxis().set_visible(False)
        i+=1

def plot_mfcc(mfcc):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)

    i = 0
    for x in range(2):
      for y in range(5):
        axes[x,y].set_title(list(mfcc.keys())[i])
        axes[x,y].imshow(list(mfcc.values())[i], cmap='hot', interpolation='nearest')
        axes[x,y].get_xaxis().set_visible(False)
        axes[x,y].get_yaxis().set_visible(False)
        i+=1

def calc_fft(y, rate):
  n = len(y)
  freq = np.fft.rfftfreq(n, d=1/rate)
  Y = abs(np.fft.rfft(y)/n)
  return Y, freq

def envelope(y, rate, threshold):
  mask = []
  y = pd.Series(y).apply(np.abs)
  y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
  for mean in y_mean:
    if mean > threshold:
      mask.append(True)
    else:
      mask.append(False)
  
  return mask

df = pd.read_csv('file.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.set_index('Filename', inplace=True)

class_dist = df.groupby(['Label'])['Length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

classes = list(np.unique(df.Label))

for c in classes:
  wav_file = df[df.Label == c].iloc[0,0]
  signal, rate = librosa.load('./wavfiles/' + wav_file, sr=44100)
  mask = envelope(signal, rate, 0.0005)
  signal = signal[mask]
  
  signals[c] = signal
  fft[c] = calc_fft(signal, rate)

  bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
  fbank[c] = bank

  mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
  mfccs[c] = mel

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbanks(fbank)
plt.show()

plot_mfcc(mfccs)
plt.show()

for filename in tqdm(df.Filename):
  signal, rate = librosa.load('./wavfiles/' + filename, sr=16000)
  mask = envelope(signal, rate, 0.0005)
  wavfile.write(filename='./clean/' + filename, rate=rate, data=signal[mask])

