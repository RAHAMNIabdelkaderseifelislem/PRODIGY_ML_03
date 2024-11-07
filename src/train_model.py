"""
Utility functions for training a model.

This module provides one function: train_model. The train_model function loads
images and labels from given paths, flattens the images, trains an SVM model
using the flattened images and labels, and saves the model to a given path.

"""
import numpy as np
from sklearn import svm
import joblib

def train_model(images_path, labels_path, output_path):
    """
    Train an SVM model using images and labels from given paths and save it to
    a given path.

    Parameters
    ----------
    images_path : str
        The path to the numpy array of images.
    labels_path : str
        The path to the numpy array of labels.
    output_path : str
        The path to the file where the model will be saved.

    Returns
    -------
    None

    """
    # Load the images and labels.
    images = np.load(images_path)
    labels = np.load(labels_path)

    # Flatten the images.
    X_train_flat = images.reshape(images.shape[0], -1)

    # Train the model.
    clf = svm.SVC()
    clf.fit(X_train_flat, labels)

    # Save the model.
    joblib.dump(clf, output_path)

