import os

import numpy as np


def select_images_iteration(directory):
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


def denormalize_stft(s):
    s_denorm = s * 3
    s_denorm *= 2.8
    s_denorm += -6
    s_denorm = np.exp(s_denorm)
    return s_denorm
