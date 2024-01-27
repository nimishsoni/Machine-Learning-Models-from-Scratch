**Naive Bayes Classifier**

This repository includes Python implementations of a Gaussian Naive Bayes Classifier, a popular probabilistic classification algorithm. The code is organized into two files: **NaiveBayes.py** and **apply_NaiveBayes.py**.

Following are the key steps for the algorithm:
- Gaussian Naive Bayes Class is fed with trained data consisting of labelled examples. 
- The labelled data is used to compute class-wise feature-mean, feature-variance and prior probabilities. The mean and variance define the probability distribution function for each feature.
- For new sample, first likelihood is computed for each feature value leveraging probability density functions
- Log of likelihood of each feature is then computed and summed up to compute posterior probability (Prior probability is also added) of data sample for class_i
- Class with highest posterior probability is selected for the data point


NaiveBayes.py

Introduction

**NaiveBayes.py** provides a Python class **GaussianNaiveBayes** for building and training a Naive Bayes classifier. This classifier is based on the assumption that features are conditionally independent given the class label, which simplifies the computation of probabilities. The code allows users to fit the model to input data and make predictions.

Dependencies

NumPy: A powerful numerical computing library in Python.
Usage

Import the **GaussianNaiveBayes** class from **NaiveBayes.py**.
Create an instance of the class.
Fit the model using the **fit** method with training data.
Make predictions on new data using the **predict** method.
Parameters

No explicit parameters are required during initialization.
Methods

**fit(X, y)**: Fit the Naive Bayes model to the input data.
**predict(X)**: Make predictions on the input data.
apply_NaiveBayes.py

**apply_NaiveBayes.py** provides an example script demonstrating how to use the **NaiveBayes** class on a synthetic dataset. The script splits the data into training and testing sets, fits the Naive Bayes model, and evaluates its classification accuracy.

Dependencies

NumPy: A powerful numerical computing library in Python.
**NaiveBayes.py**: The **NaiveBayes** class.
Usage

Import required libraries and the **NaiveBayes** class.
Generate a synthetic dataset or use your own data.
Split the dataset into training and testing sets.
Create an instance of the **NaiveBayes** class.
Fit the model using the training data.
Make predictions on the test data and evaluate the classification accuracy.
Feel free to explore the code, experiment with different datasets, and customize as needed. If you have any questions or need further assistance, please don't hesitate to reach out.