# Hindi-OCR

This  model involves recognition of hindi hand written characters using Convolutional neural network. 
Python implementation using keras has been done. The dataset can be downloaded from the following link:
https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

# Dataset
The dataset contains 92,000 images of handwritten hindi characters belonging to 46 classes. The data is splitted into training set(85%) and test set(15%). The images are of size 32x32 in .png format.

# Architecture
The architecture involves four CNN layers followed by 3 fully connected layer. The final layers using softmax function helps to provide the most probable answer.

# Implementation Details
Loss function- Categorical Cross entropy
Optimizer- Adam
Final output layer activation- Softmax

Model Train file- Hindi.py
Testing individual images- check.py

# Output
Accuracy achieved in 25 epochs- 98.94%
