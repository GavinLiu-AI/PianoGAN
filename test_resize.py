import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import skimage.transform
import cv2


DATASET_DIR = "./data/maestro-v3.0.0"
AUDIO_CHUNKS_DIR = "./data/audio_chunks/"
SPECTROGRAM_DIR = "./data/spectrograms/"
AUDIO_OUT_DIR = "./output/"
STFT_ARRAY_DIR = "./data/stft_arrays/"
hyperparams = {"sampling_rates": [22050],
               "n_ffts": [2048],  # hop lengths = None
               "betas": [0.98],
               "eps": [1e-7]
               }
downsample_sizes = [(50, 50), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500), (600, 600)]


def get_dataset_paths(directory, extension):
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(subdir, file)
                print("Adding: ", path)
                paths.append(path)
    return paths


def make_audio_chunks(paths, seconds=20):
    for audio_path in paths:
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = seconds*1000  # milli seconds
        chunks = make_chunks(audio, chunk_length_ms)
        chunks.pop(-1)

        # Export all of the individual chunks as wav files
        for i, chunk in enumerate(chunks):
            _, chunk_name = os.path.split(os.path.splitext(audio_path)[0] + "_chunk_{0}.wav".format(i))
            print("Exporting ", chunk_name)
            chunk.export(AUDIO_CHUNKS_DIR + chunk_name, format="wav")

    print("\n\nChunks export completed.")


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


def scale_min_max(x, minimum, maximum):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (maximum - minimum) + minimum
    return x_scaled


def convert_audio_to_arrays(sampling_rate=22050, n_fft=2048, divide_time=2, divide_freq=2, eps=1e-7):
    paths = get_dataset_paths("./data/test_chunks/", ".wav")
    mag_mean = 0
    mag_std = 0
    name = str(sampling_rate) + '_' + str(n_fft)
    for path in paths:
        if path[19: 21] != '._':
            print("Converting ", path)

            y, sr = librosa.load(path, sr=sampling_rate)

            # Decompose a spectrogram with NMF
            D = librosa.stft(y, n_fft=n_fft, hop_length=None)

            # Separate the magnitude and phase and only use magnitude
            S, _ = librosa.magphase(D)

            # Processing step for the magnitude matrix of the STFT.
            # Take the logarithm of the magnitudes, normalize it, clip it at 3*std and rescale to [-1,1]
            S = np.log(S)
            # normalizing and storing exponentially weighted averages for mean and standard deviation
            mag_mean = np.mean(S)  # (mag_mean*beta + (1-beta) * np.mean(S))/(1-beta**t)
            mag_std = np.std(S)  # (mag_std*beta + (1-beta) * np.std(S))/(1-beta**t)
            S = (S - mag_mean) / (mag_std + eps)
            # clipping
            S = np.where(np.abs(S) < 3 * mag_std, S, 3 * mag_std * np.sign(S))
            # rescale to [-1,1]
            S /= 3

            out = "./data/test_stft/" + name + ".npy"
            np.save(out, S)

            for size in downsample_sizes:
                downsample(S, size)

    return {'Mean Magnitude': mag_mean, 'Std Magnitude': mag_std, 'sampling_rate': sampling_rate,
            'n_fft': n_fft, 'divide_time': divide_time,
            'divide_freq': divide_freq, 'n_fft': n_fft, 'eps': eps}


def downsample(S, size):
    S_downsample = skimage.transform.resize(S, size, anti_aliasing=True)
    out = "./data/test_stft_downsample/" + str(size) + ".npy"
    np.save(out, S_downsample)


def upsample(S_downsample, size):
    S_upsample = cv2.resize(S_downsample, (862, 1025), interpolation=cv2.INTER_NEAREST)
    out = "./data/test_stft_upsample/" + str(size) + "_repeats" + ".npy"
    np.save(out, S_upsample)
    S_upsample = cv2.resize(S_downsample, (862, 1025), interpolation=cv2.INTER_LINEAR)
    out = "./data/test_stft_upsample/" + str(size) + "_linear" + ".npy"
    np.save(out, S_upsample)
    S_upsample = cv2.resize(S_downsample, (862, 1025), interpolation=cv2.INTER_AREA)
    out = "./data/test_stft_upsample/" + str(size) + "_area" + ".npy"
    np.save(out, S_upsample)


def run_reconstruction(cache):
    paths = get_dataset_paths('./data/test_stft_downsample/', ".npy")
    for path in paths:
        for size in downsample_sizes:
            if str(size) in path:
                audio_reconstruction(cache, path, size)


def audio_reconstruction(cache, path, size):
    mag_mean = cache['Mean Magnitude']
    mag_std = cache['Std Magnitude']
    sampling_rate = cache['sampling_rate']
    eps = cache['eps']
    print("Reconstructing: ", path)
    S = np.load(path)
    upsample(S, size)
    paths = []
    paths.append("./data/test_stft_upsample/" + str(size) + "_repeats" + ".npy")
    paths.append("./data/test_stft_upsample/" + str(size) + "_linear" + ".npy")
    paths.append("./data/test_stft_upsample/" + str(size) + "_area" + ".npy")
    for path in paths:
        # back to array
        S = np.load(path)
        S = S * 3
        S = S * (mag_std + eps) + mag_mean
        S = np.exp(S)

        y = librosa.griffinlim(S)

        _, file_name = os.path.split(path)
        out = AUDIO_OUT_DIR + os.path.splitext(file_name)[0] + ".wav"

        # Save reconstructed data
        scipy.io.wavfile.write(out, sampling_rate, y)


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


if __name__ == "__main__":
    cache = convert_audio_to_arrays(sampling_rate=22050, n_fft=2048, divide_time=2, divide_freq=2, eps=1e-7)
    run_reconstruction(cache)
    # record_mean_std()
