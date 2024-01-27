**PCA (Principal Component Analysis)**

This repository contains a Python implementation of Principal Component Analysis (PCA), a dimensionality reduction technique widely used in data science and machine learning.

In PCA algorithm, we are interested in finding the principal components, which are directions in feature space along which the variation in the dataset is the most. 

These principal components are eigen vectors of covariance matrix and their corresponding eigen values are magnitude of variance along those respective directions. 

Eigen vectors with top k eigen values are selected as principal components. The goal is to find these principal components explaining most of the variations and reduce dimensionality of feature space. 

**PCA.py**
Introduction

PCA.py provides a Python class PCA for performing Principal Component Analysis. The code allows users to fit the model to input data and transform data into the reduced-dimensional space defined by the principal components.
Dependencies

**NumPy**: A powerful numerical computing library in Python.
Usage

Import the PCA class from PCA.py.
Create an instance of the class with the desired number of principal components.
Fit the model using the fit method with training data.
Transform data using the transform method to obtain reduced-dimensional representations.
Parameters

n_components (int): Number of principal components to retain.
Methods

fit(X): Fit the PCA model to the input data.
transform(X): Transform input data into the reduced-dimensional space.
apply_PCA.py

apply_PCA.py provides an example script demonstrating how to use the PCA class on the Iris dataset. The script visualizes the data in the reduced-dimensional space defined by the top principal components.
Dependencies

NumPy: A powerful numerical computing library in Python.
PCA.py: The PCA class.

**Usage**
- Import required libraries and the PCA class.
- Load the Iris dataset using scikit-learn.
- Create an instance of the PCA class with the desired number of principal components.
- Fit the model using the dataset.
- Transform the data and visualize the results using matplotlib.