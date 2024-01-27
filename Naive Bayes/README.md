**Naive Bayes Classifier**

This repository includes Python implementations of both Gaussian Naive Bayes and Multinomial Naive Bayes classifiers, popular probabilistic classification algorithms. The code is organized into two files: **NaiveBayes.py** and **apply_NaiveBayes.py**.

Following are the key steps for the algorithms:

Gaussian Naive Bayes:

Gaussian Naive Bayes Class is fed with trained data consisting of labelled examples.
The labelled data is used to compute class-wise feature mean, feature variance, and prior probabilities. The mean and variance define the probability distribution function for each feature.
For a new sample, the likelihood is computed for each feature value leveraging probability density functions.
The log of the likelihood of each feature is then computed and summed up to compute the posterior probability (prior probability is also added) of the data sample for class_i.
The class with the highest posterior probability is selected for the data point.
Multinomial Naive Bayes:

Multinomial Naive Bayes Class is fed with trained data consisting of labelled examples.
The class-wise feature counts and class counts are calculated based on the training data.
For a new sample, log-likelihoods are computed for each class based on the multinomial distribution.
The class with the highest log-likelihood is selected for the data point.
**NaiveBayes.py**

Introduction

NaiveBayes.py provides Python classes GaussianNaiveBayes and MultinomialNaiveBayes for building and training Naive Bayes classifiers. These classifiers are based on the assumption that features are conditionally independent given the class label, simplifying the computation of probabilities. The code allows users to fit the models to input data and make predictions.

Dependencies

NumPy: A powerful numerical computing library in Python.
Usage

Import the GaussianNaiveBayes or MultinomialNaiveBayes class from NaiveBayes.py.
Create an instance of the class.
Fit the model using the fit method with training data.
Make predictions on new data using the predict method.
For Gaussian Naive Bayes, visualize Probability Distribution Functions using the plot_pdf method.
Parameters

No explicit parameters are required during initialization for both Gaussian and Multinomial Naive Bayes.

Methods

**GaussianNaiveBayes**:

fit(X, y): Fit the Gaussian Naive Bayes model to the input data.
predict(X): Make predictions on the input data.
plot_pdf(feature_index): Plot the Gaussian Probability Distribution Functions for each class for a specific feature.
MultinomialNaiveBayes:

fit(X, y): Fit the Multinomial Naive Bayes model to the input data.
predict(X): Make predictions on the input data.
apply_NaiveBayes.py

Introduction

apply_NaiveBayes.py provides an example script demonstrating how to use both the Gaussian and Multinomial NaiveBayes classes on a synthetic dataset. The script splits the data into training and testing sets, fits the Naive Bayes models, and evaluates their classification accuracies.

**Dependencies**

NumPy: A powerful numerical computing library in Python.

**NaiveBayes.py**: The GaussianNaiveBayes and MultinomialNaiveBayes classes.

**Usage**

- Import required libraries and the GaussianNaiveBayes and MultinomialNaiveBayes classes.
- Generate a synthetic dataset or use your own data.
- Split the dataset into training and testing sets.
- Create instances of the GaussianNaiveBayes and MultinomialNaiveBayes classes.
- Fit the models using the training data.
- Make predictions on the test data and evaluate the classification accuracies.
- For Gaussian Naive Bayes, visualize Probability Distribution Functions for specific features.

Feel free to explore the code, experiment with different datasets, and customize as needed. If you have any questions or need further assistance, please don't hesitate to reach out.