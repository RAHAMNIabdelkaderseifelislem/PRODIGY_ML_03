"""
This module provides a GUI for classifying images as 'Cat' or 'Dog' using a pre-trained SVM model.

The GUI is built using the Gradio library, which allows users to upload an image and receive a classification result.

Functions:
- classify_image: Classifies an image using a saved SVM model.
- create_gui: Creates and launches the Gradio interface for image classification.
"""

import gradio as gr
from src.predict import predict_image

def classify_image(image):
    """
    Classify an image as 'Cat' or 'Dog' using a saved SVM model.

    Parameters
    ----------
    image : str
        The file path to the image to classify.

    Returns
    -------
    str
        The predicted class ('Cat' or 'Dog').
    """
    # Predict the class of the image using the SVM model.
    prediction = predict_image(image, 'models/svm.pkl')
    return prediction

def create_gui():
    """
    Create and launch the Gradio interface for image classification.

    This function sets up the input and output interfaces for the Gradio app
    and launches it in a web browser.
    """
    # Define the input as an image file path and the output as a label.
    inputs = gr.inputs.Image(type='filepath')
    outputs = gr.outputs.Label(num_top_classes=1)
    
    # Create the Gradio interface and launch it.
    interface = gr.Interface(fn=classify_image, inputs=inputs, outputs=outputs)
    interface.launch()

