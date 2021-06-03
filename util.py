import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import librosa.display

eps = 1e-7

df = pd.read_csv("./data/saved_mean_std.csv")
means = df['mean']
stds = df['std']
mag_mean = 0
for index in range(means.shape[0]):
    if means[index] != -np.inf:
        mag_mean += means[index] / means.shape[0]
mag_std = np.mean(stds)


def plot_loss(path):
    df = pd.read_pickle(path)
    epoch = df['epoch']
    d_loss = df['d_loss']
    d_acc = df['d_acc']
    g_loss = df['g_loss']

    fig, ax = plt.subplots()
    ax.set_xlabel('num_epochs')
    ax.set_ylabel('loss')
    ax.plot(epoch, d_loss, c='b', label='d_loss')
    ax.plot(epoch, d_acc, c='g', label='d_acc')
    ax.plot(epoch, g_loss, c='r', label='g_loss')
    ax.legend()
    plt.savefig("./loss_plot.png")


def get_dataset_paths(directory, extension):
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(subdir, file)
                paths.append(path)
    return paths


def upsample(S_downsample):
    return cv2.resize(S_downsample, (862, 1025), interpolation=cv2.INTER_LINEAR)


def audio_reconstruction(S):
    S = upsample(np.squeeze(S))
    S = S * 3
    S = S * (mag_std + eps) + mag_mean
    S = np.exp(S)

    return librosa.griffinlim(S)