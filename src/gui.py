"""
This module provides a function to create a GUI for classifying new images using a saved SVM model.

The GUI is created using the Gradio library and allows the user to select an image file and get a prediction of the class of the image.

"""

import gradio as gr
from src.predict import predict_image


def classify_image(image):
    """
    This function takes an image file and uses the predict_image function to get a prediction of the class of the image.

    Parameters
    ----------
    image : str
        The path to the image file.

    Returns
    -------
    str
        The predicted class of the image.
    """
    prediction = predict_image(image.name, 'models/svm.pkl')
    return prediction


def create_gui():
    """
    This function creates a Gradio interface for classifying new images.

    The interface takes an image file as input and outputs the predicted class of the image.
    """
    # Updated to use current Gradio API
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="filepath"),  # Direct usage instead of gr.inputs
        outputs=gr.Label(num_top_classes=1)  # Direct usage instead of gr.outputs
    )
    interface.launch()

