# Cat vs Dog Classifier 🐕‍🦺🐈

This is a simple cat vs dog classifier that uses a Support Vector Machine (SVM) to classify images of cats and dogs. The model is trained on a dataset of 25,000 images of cats and dogs, and the accuracy of the model on the testing data is 90%.

## How to run the project

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Download the dataset from Kaggle and extract it to the `data` directory.
4. Run the `main.py` script to preprocess the data, train the model, and start the GUI.

## Project structure

- `data/`: Directory containing the dataset of cat and dog images.
  - `train/`: Directory containing the training images.
  - `test/`: Directory containing the testing images.
- `src/`: Directory containing the source code.
  - `preprocess.py`: Module for preprocessing the images.
  - `train_model.py`: Module for training the SVM classifier.
  - `predict.py`: Module for making predictions on new images.
  - `gui.py`: Module for creating the Gradio-based GUI.
- `models/`: Directory containing the trained SVM model.
- `main.py`: Main script that calls the functions in the `src` modules to preprocess the data, train the model, and start the GUI.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Results

The accuracy of the model on the testing data is 90%.

## Acknowledgements

This project is Task 3 of the Prodigy InfoTech ML internship.
