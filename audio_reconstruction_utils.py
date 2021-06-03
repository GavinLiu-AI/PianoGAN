import os

import numpy as np


def select_images_iteration(directory):
    """
    Helper function used to select which folder of generated images needs to be processed

    :param directory: folder that stores generated images

    :returns absolute path of the selected folder
    """
    sub_dirs = next(os.walk(directory))[1]
    for index, sub_dir in enumerate(sub_dirs):
        print("{}. {}".format(index + 1, sub_dir))

    while True:
        try:
            select = int(input("\nSelect folder to reconstruct: "))
        except ValueError:
            print("Invalid.")
            continue
        else:
            if 0 < select <= len(sub_dirs):
                dir_selected = sub_dirs[select - 1]
                break
            else:
                print("Invalid.")
                continue

    return directory + dir_selected + "/", sub_dir + "/"


def unnormalize_stft(s):
    """
    Revert normalization on STFT matrix

    :param s: STFT matrix

    :return: STFT matrix post de-normalization
    """
    s_denorm = s * 3
    s_denorm *= 2.8
    s_denorm += -6
    s_denorm = np.exp(s_denorm)
    return s_denorm


def decrease_brightness(s_scaled):
    """
    Decrease pixel brightness for better audio quality

    :param s_scaled: STFT matrix

    :return: STFT matrix with decreased pixel brightness
    """
    s_scaled[s_scaled <= 0] = 0
    s = 255 * (s_scaled - np.min(s_scaled)) / (np.max(s_scaled) - np.min(s_scaled))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if 20 < s[i][j] < 200:
                s[i][j] /= 10
    return s
