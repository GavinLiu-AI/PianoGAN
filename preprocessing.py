import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage.io
import scipy.io.wavfile
import skimage.transform
import pandas as pd


DATASET_DIR = "./data/maestro-v3.0.0"
AUDIO_CHUNKS_DIR = "./data/audio_chunks/"
SPECTROGRAM_DIR = "./data/spectrograms/"
AUDIO_OUT_DIR = "./output/"
STFT_ARRAY_DIR = "./data/stft_arrays/"
PROCESSED_STFT_DIR = "./data/clipped_stft/"
RESIZED_STFT_DIR = "./data/resized_stft/"


# retrieve all paths to files in a specific folder with certain extension
def get_dataset_paths(directory, extension):
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(subdir, file)
                print("Adding: ", path)
                paths.append(path)
    return paths


# crop audio files into 20 seconds chunks
def make_audio_chunks(paths):
    for audio_path in paths:
        if audio_path[27: 29] != "._":
            audio = AudioSegment.from_file(audio_path)
            chunk_length_ms = 20000  # 20 seconds
            chunks = make_chunks(audio, chunk_length_ms)
            chunks.pop(-1)

            # Export all of the individual chunks as wav files
            for i, chunk in enumerate(chunks):
                _, chunk_name = os.path.split(os.path.splitext(audio_path)[0] + "_chunk_{0}.wav".format(i))
                print("Exporting ", chunk_name)
                chunk.export(AUDIO_CHUNKS_DIR + chunk_name, format="wav")

    print("\n\nChunks export completed.")


# generate spectrograms from audio files
def display_spectrogram():
    paths = get_dataset_paths(AUDIO_CHUNKS_DIR, ".wav")[:3]

    for path in paths:
        print("Converting ", path)

        y, sr = librosa.load(path)

        # Decompose a spectrogram with NMF
        # Short-time Fourier transform underlies most analysis.
        # librosa.stft returns a complex matrix D.
        # D[f, t] is the FFT value at frequency f, time (frame) t.
        D = librosa.stft(y)

        # Separate the magnitude and phase and only use magnitude
        S, phase = librosa.magphase(D)
        print("S Shape: ", S.shape)

        melspec_log = librosa.feature.melspectrogram(S=np.log(S), sr=sr)
        print("MelSpec Shape: ", melspec_log.shape)

        plt.figure()
        librosa.display.specshow(melspec_log, y_axis='mel', x_axis='time')
        plt.colorbar()
        plt.show()


# use STFT to convert audio files to arrays
def convert_audio_to_arrays():
    paths = get_dataset_paths(AUDIO_CHUNKS_DIR, ".wav")

    tic = time.time()
    for path in paths:
        print("Converting ", path)

        y, sr = librosa.load(path)

        # Decompose a spectrogram with NMF
        D = librosa.stft(y)

        # Separate the magnitude and phase and only use magnitude
        S, phase = librosa.magphase(D)  # S shape (1025, 862)

        _, file_name = os.path.split(path)
        out = STFT_ARRAY_DIR + os.path.splitext(file_name)[0] + ".npy"
        np.save(out, S)

    print("\nTotal time: ", time.time() - tic)


# reconstruct arrays into audio clips
def audio_reconstruction():
    paths = get_dataset_paths(STFT_ARRAY_DIR, ".npy")

    for path in paths:
        print("Reconstructing: ", path)
        S = np.load(path)
        y = librosa.griffinlim(S)

        _, file_name = os.path.split(path)
        out = AUDIO_OUT_DIR + os.path.splitext(file_name)[0] + ".wav"

        # Save reconstructed data
        scipy.io.wavfile.write(out, 22050, y)


# process all arrays to record mean and std for later use
def record_mean_std():
    paths = get_dataset_paths(STFT_ARRAY_DIR, ".npy")

    mean_list = []
    std_list = []

    for path in paths:
        S = np.load(path)
        S = np.log(S)
        mag_mean = np.mean(S)
        mag_std = np.std(S)
        mean_list.append(mag_mean)
        std_list.append(mag_std)
        print("Finished:", path)

    data = {"mean": mean_list, "std": std_list, "path": paths}
    df = pd.DataFrame.from_dict(data)
    df.to_csv("./data/saved_mean_std.csv")


# downsample all arrays in the dataset
def preprocessing_arrays():
    df = pd.read_csv("./data/saved_mean_std.csv")
    paths = df['path']
    means = df['mean']
    stds = df['std']
    eps = 1e-7

    for index in range(len(paths)):
        print("Processing: ", paths[index])
        S = np.load(paths[index])

        # Processing step for the magnitude matrix of the STFT.
        # Take the logarithm of the magnitudes, normalize it, clip it at 3*std and rescale to [-1,1]
        S = np.log(S)
        S = (S - means[index]) / (stds[index] + eps)
        # clipping
        S = np.where(np.abs(S) < 3, S, 3 * np.sign(S))
        # rescale to [-1,1]
        S /= 3

        _, file_name = os.path.split(paths[index])
        out = PROCESSED_STFT_DIR + os.path.splitext(file_name)[0] + ".npy"
        np.save(out, S)


# function to downsample all arrays and save to folder
def downsample():
    paths = get_dataset_paths(PROCESSED_STFT_DIR, ".npy")
    for path in paths:
        S = np.load(path)
        S_downsample = skimage.transform.resize(S, (256, 256), anti_aliasing=True)
        _, file_name = os.path.split(path)
        out = RESIZED_STFT_DIR + os.path.splitext(file_name)[0] + ".npy"
        np.save(out, S_downsample)


if __name__ == "__main__":
    audio_paths = get_dataset_paths(DATASET_DIR, ".wav")
    make_audio_chunks(audio_paths)
    display_spectrogram()
    convert_audio_to_arrays()
    # audio_reconstruction()
    record_mean_std()
    preprocessing_arrays()
    downsample()
