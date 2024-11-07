"""
Utility functions for predicting the class of an image using a saved SVM model.

This module provides one function: predict_image. The predict_image function
loads an image from a given path, resizes it to a given size, normalizes it,
loads an SVM model from a given path, and uses the model to predict the class
of the image.

"""

import cv2
import numpy as np
import joblib


def predict_image(image_path, model_path):
    """
    Predict the class of an image using a saved SVM model.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    model_path : str
        The path to the file where the SVM model is saved.

    Returns
    -------
    str
        The predicted class ('Cat' or 'Dog').

    """
    # Load the image.
    img = cv2.imread(image_path)

    # Resize the image to 64x64.
    img = cv2.resize(img, (64, 64))

    # Normalize the image.
    img = img / 255.0

    # Reshape the image to a 1D array.
    img_flat = img.reshape(1, -1)

    # Load the SVM model.
    clf = joblib.load(model_path)

    # Predict the class of the image.
    prediction = clf.predict(img_flat)

    # Return the predicted class.
    return 'Cat' if prediction[0] == 0 else 'Dog'

