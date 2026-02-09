# MNIST Digit Classifier using TensorFlow

This project implements a handwritten digit classifier using the MNIST dataset and TensorFlow.
In addition to evaluating the model on the MNIST test dataset, the project also supports
prediction of **manually handwritten digits written on paper**, after appropriate image preprocessing.


## Project Overview

The objectives of this project are to:
- Build and train a neural network using the MNIST dataset
- Evaluate its performance on unseen test data
- Save the trained model for reuse
- Predict digits from **real-world handwritten images provided by the user**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits:
- Image size: 28 × 28 pixels
- Labels: digits from 0 to 9

Split:
- Training set: 60,000 images
- Test set: 10,000 images

---

## Model Architecture

The model is a feedforward neural network built using the Keras Sequential API:
- Flatten layer to convert images into vectors
- Dense layer with 128 neurons and ReLU activation
- Dense layer with 64 neurons and ReLU activation
- Output layer with 10 neurons and Softmax activation

## Training Details

- Framework: TensorFlow (Keras)
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 64
- Epochs: 5
- Validation Split: 10% of training data

Validation data is used during training to monitor performance and detect overfitting.
The test dataset is used only once for final evaluation.

## Results

- Test Accuracy on MNIST: 97.58%
- Successfully predicts manually handwritten digits after preprocessing


## Installation
Go to the folder where your project is saved using cd then 
'''bash'''
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Training
Train the model using python train.py
This will give you the output on the basis of the MNIST default dataset of 70,000 images.
now, use python predict.py, this will predict the number uploaded.


