import datetime
import os
import time


def get_filename(path):
    """
    Function used to extract filename from an absolute path

    :param path: absolute path

    :return: filename
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_filename_list(paths):
    """
    Function used to retrieve a list of filenames from a list of absolute paths

    :param paths: list of absolute file paths

    :return: list of filenames
    """
    return [os.path.splitext(os.path.basename(path))[0] for path in paths]


def get_unprocessed_items(src_dir, dest_dir):
    """
    Function used to identify and retrieve a list of unprocessed files

    :param src_dir: source directory of files need to be processed
    :param dest_dir: output directory

    :return: list of unprocessed files
    """
    print("\nEvaluating unprocessed files in {}".format(src_dir))

    src_paths = get_absolute_file_paths(src_dir)
    dest_paths = get_absolute_file_paths(dest_dir)

    if src_paths and len(src_paths) != len(dest_paths):
        src_filenames = get_filename_list(src_paths)
        dest_filenames = get_filename_list(dest_paths)
        src_ext = os.path.splitext(os.path.basename(src_paths[0]))[1]
        return [src_dir + file + src_ext for file in src_filenames if file not in dest_filenames]
    else:
        return []


def get_absolute_file_paths(directory, extension=None):
    """
    Function used to retrieve absolute paths to all files of specified file type under a directory

    :param directory: the directory to perform search and retrieve
    :param extension: specified file type, defaulted to None. The function will return paths to all files with no file
    type is specified

    :return: list of paths of all files that matches the file type under directory
    """
    print("\nRetrieving absolute paths from {}".format(directory))
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(subdir, file)
            if extension:
                if file.lower().endswith(extension):
                    paths.append(path)
            else:
                paths.append(path)
    return paths


def display_progress_eta(current_item, total_items, start_time):
    """
    Function used to display progress and ETA on the console.

    :param current_item: current item being processed
    :param total_items: list of items to be processed

    :param start_time: start time of the entire process
    """
    index = total_items.index(current_item) + 1
    len_total = len(total_items)
    percentage = round(index / len_total * 100)
    seconds = (time.time() - start_time) / index * (len_total - index)
    eta = str(datetime.timedelta(seconds=round(seconds)))
    print("\r\nProcessed {}% ({}/{}) \nETA: {}".format(percentage, index, len_total, eta))
