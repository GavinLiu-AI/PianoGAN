import os
import time

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile

import audio_reconstruction_utils as ar_utils
import preprocessing as prep
import preprocessing_utils as prep_utils


def audio_reconstruction_stylegan(src_dir, dest_dir, resize_h, resize_w):
    src_dir, sub_dir = ar_utils.select_images_iteration(directory=src_dir)
    paths = prep_utils.get_absolute_file_paths(src_dir)

    out_dir = dest_dir + sub_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)

        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        S_recovered = np.array(image_gray, dtype=np.float32)
        S_recovered = cv2.resize(S_recovered, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        S = (S_recovered - np.min(S_recovered)) / (np.max(S_recovered) - np.min(S_recovered)) * 2 - 1
        pd.DataFrame(S).to_csv(out_dir + prep_utils.get_filename(path) + "_norm.csv", header=None, index=False)
        plt.imsave(out_dir + prep_utils.get_filename(path) + "_norm.png", S)

        S = ar_utils.denormalize_stft(s=S)
        pd.DataFrame(S).to_csv(out_dir + prep_utils.get_filename(path) + "_reconstruct.csv", header=None, index=False)
        plt.imsave(out_dir + prep_utils.get_filename(path) + "_reconstruct.png", S)

        y = librosa.griffinlim(S)

        out = out_dir + prep_utils.get_filename(path) + ".wav"

        scipy.io.wavfile.write(out, 22050, y)


def organize_temp_fake_images(src_dir, dest_dir):
    if os.listdir(src_dir):
        print("\nFound new temp images, moving them to stft_image_fake folders.")
        new_dir = dest_dir + time.strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(new_dir)

        for file in os.listdir(src_dir):
            os.rename(src_dir + file, new_dir + file)


def audio_reconstruction_test(src_dir, dest_dir, ext=".png", size=None):
    paths = prep_utils.get_unprocessed_items(src_dir=src_dir, dest_dir=dest_dir)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)

        S = np.load(path)
        cv2.imshow("image", S)
        cv2.waitKey(0)
        pd.DataFrame(S).to_csv(dest_dir + "S.csv", header=None, index=False)

        S_scaled = prep_utils.scale(S)
        cv2.imshow("image", S_scaled)
        cv2.waitKey(0)
        pd.DataFrame(S_scaled).to_csv(dest_dir + "S_scaled.csv", header=None, index=False)

        out_path = dest_dir + "gray" + ext
        cv2.imwrite(out_path, S_scaled)

        S = cv2.imread(out_path, 0)
        S = np.array(S, dtype=np.float32)
        S_recovered = S

        if size:
            S_recovered = cv2.resize(S_recovered, (size, size), interpolation=cv2.INTER_CUBIC)
        pd.DataFrame(S_recovered).to_csv(dest_dir + "S_recovered.csv", header=None, index=False)
        out_path = dest_dir + "resized" + ext
        cv2.imwrite(out_path, S_recovered)

        S_audio = np.genfromtxt(dest_dir + "S_recovered.csv", delimiter=',')
        S_audio = np.array(S_audio, dtype=np.float32)
        S_audio = cv2.resize(S_audio, (431, 1025), interpolation=cv2.INTER_CUBIC)

        y = librosa.griffinlim(S_audio)

        out = dest_dir + "s.wav"

        # Save reconstructed data
        scipy.io.wavfile.write(out, 22050, y)
        break


if __name__ == "__main__":
    # organize_temp_fake_images(src_dir=prep.STYLEGAN_STFT_IMAGES_FAKE_TEMP_DIR,
    #                           dest_dir=prep.STYLEGAN_STFT_IMAGES_FAKE_DIR)
    # audio_reconstruction_stylegan(src_dir=prep.STYLEGAN_STFT_IMAGES_FAKE_DIR, dest_dir=prep.STYLEGAN_AUDIO_OUTPUT_DIR,
    #                               resize_h=1025, resize_w=431)
    audio_reconstruction_test(src_dir=prep.STYLEGAN_STFT_ARRAYS_DIR, dest_dir=prep.STYLEGAN_AUDIO_TEST_DIR, size=512)
