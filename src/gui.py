"""
This module provides a function to create a GUI for classifying new images using a saved SVM model.

The GUI is created using the Gradio library and allows the user to select an image file and get a prediction of the class of the image.
"""

import gradio as gr
from src.predict import predict_image


def classify_image(image_path):
    """
    This function takes an image path and uses the predict_image function to get a prediction of the class of the image.

    Parameters
    ----------
    image_path : str
        The path to the image file.

    Returns
    -------
    str
        The predicted class of the image.
    """
    # Now we can use the image_path directly since it's a string
    prediction = predict_image(image_path, 'models/svm.pkl')
    return prediction


def create_gui():
    """
    This function creates a Gradio interface for classifying new images.

    The interface takes an image file as input and outputs the predicted class of the image.
    """
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="filepath"),
        outputs=gr.Label(),
        title="Cat vs Dog Classifier",
        description="Upload an image of a cat or dog to classify it."
    )
    interface.launch()
