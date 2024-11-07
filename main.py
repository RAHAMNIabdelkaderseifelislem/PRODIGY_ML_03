"""
This script preprocesses image data, trains an SVM model for classifying images as 'Cat' or 'Dog', and launches a GUI for image classification.

The script performs the following steps:
1. Loads and preprocesses training images.
2. Saves the preprocessed images.
3. Trains an SVM model using the preprocessed images and associated labels.
4. Launches a GUI for classifying new images using the trained model.
"""

from src.preprocess import load_images, save_data
from src.train_model import train_model
from src.gui import create_gui
import os

if __name__ == '__main__':

    # step 0: create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and preprocess the training images
    train_images = load_images('data/train')
    
    # Step 2: Save the preprocessed images
    save_data(train_images, 'data/processed')

    # Step 3: Train the SVM model using the preprocessed images and labels
    train_model('data/processed/images.npy', 'data/processed/labels.npy', 'models/svm.pkl')

    # Step 4: Launch the GUI for image classification
    create_gui()

