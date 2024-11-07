"""
This script is the main entry point for the Cat vs Dog classification project.

It provides a command line interface for preprocessing the training data,
training a model, and starting a GUI for classifying new images.

"""

import os
from src.preprocess import load_images, save_data
from src.train_model import train_model
from src.gui import create_gui

def main():
    """
    The main function.

    This function preprocesses the training data, trains a model, and starts a
    GUI for classifying new images.
    """
    # Create the data/processed directory if it does not exist
    os.makedirs('data/processed', exist_ok=True)

    # Preprocess the training data
    train_images, train_labels = load_images('data/train')
    save_data(train_images, train_labels, 'data/processed')

    # Train the model
    train_model('data/processed/images.npy', 'data/processed/labels.npy', 'models/svm.pkl')

    # Start the GUI
    create_gui()

if __name__ == '__main__':
    main()

