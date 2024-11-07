"""
Utility functions for preprocessing the training data.

This module provides two functions: load_images and save_data.

The load_images function loads images from a given directory, resizes them to
64x64, normalizes them, and returns them as a numpy array along with their
labels.

The save_data function saves the images and labels to a given directory as
numpy arrays.

"""

import os
import cv2
import numpy as np


def load_images(data_path):
    """
    Load images from a given directory, resize them to 64x64, normalize them,
    and return them as a numpy array along with their labels.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the images.

    Returns
    -------
    tuple
        A tuple containing two elements: a numpy array of images and a numpy
        array of labels.
    """
    images = []
    labels = []
    for filename in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, filename))
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        images.append(img)
        label = filename.split('.')[0]
        label = 0 if label == 'cat' else 1
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def save_data(images, labels, output_path):
    """
    Save the images and labels to a given directory as numpy arrays.

    Parameters
    ----------
    images : numpy array
        The array of images.
    labels : numpy array
        The array of labels.
    output_path : str
        The path to the directory where the data will be saved.
    """
    np.save(os.path.join(output_path, 'images.npy'), images)
    np.save(os.path.join(output_path, 'labels.npy'), labels)

