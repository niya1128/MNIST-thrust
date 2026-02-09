import cv2
import numpy as np
import tensorflow as tf
from model import build_model

# Load trained model
model = build_model()
model.load_weights("mnist_model.weights.h5")

# Load image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# Invert colors (white bg → black)
img = cv2.bitwise_not(img)

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 28, 28)

# Predict
prediction = model.predict(img)
digit = np.argmax(prediction)

print("Predicted digit:", digit)
