"""
Utility functions for loading and saving images.

This module provides two functions: load_images and save_data. The load_images
function loads images from a given directory, resizes them to a given size, and
normalizes them. The save_data function saves an array of images to a given
directory.

"""
import os
import cv2
import numpy as np

def load_images(data_path):
    """
    Load images from a given directory, resize them to a given size, and normalize them.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the images.

    Returns
    -------
    images : numpy.ndarray
        An array of images, where each image is a 3D numpy array of size
        (height, width, 3).

    """
    images = []
    for filename in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, filename))
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        images.append(img)
    images = np.array(images)
    return images


def save_data(images, output_path):
    """
    Save an array of images to a given directory.

    Parameters
    ----------
    images : numpy.ndarray
        An array of images, where each image is a 3D numpy array of size
        (height, width, 3).

    output_path : str
        The path to the directory where the images will be saved.

    Returns
    -------
    None

    """
    np.save(os.path.join(output_path, 'images.npy'), images)

