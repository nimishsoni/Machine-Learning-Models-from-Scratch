**Logistic Regression**
- This repository contains a simple implementation of Logistic Regression in Python. Logistic Regression is a supervised machine learning algorithm used for binary classification problems.

**LogisticRegression.py**
**Introduction**
- LogisticRegression.py contains the implementation of the Logistic Regression classifier. The code provides flexibility with options for regularization, learning rate decay, early stopping, and patience.

**Dependencies**
- NumPy: A powerful numerical computing library in Python.

**Usage**
- Import the LogisticRegression class from LogisticRegression.py.
- Create an instance of the class with desired parameters.
- Fit the model using the fit method with training data.
- Predict using the predict method on new data.

**Parameters**
- lr (float): Learning rate for gradient descent.
- decay_rate (float): Rate at which the learning rate decays.
- n_iters (int): Number of iterations for training.
- regularization (str): Type of regularization, 'l1', 'l2', or None.
- alpha (float): Regularization strength.
- early_stopping (bool): Enable early stopping.
- patience (int): Number of iterations to wait for improvement during early stopping.

**Methods**
- fit(X, y, val_set=None): Train the logistic regression model.
- predict(X): Make predictions using the trained model.

apply_LogisticRegression.py
- apply_LogisticRegression.py provides an example of how to use the Logistic Regression class on a real-world dataset. In this case, the Breast  Cancer dataset from the scikit-learn library is used.

**Dependencies**
- NumPy: A powerful numerical computing library in Python.
- LogisticRegression.py: The Logistic Regression class.

**Usage**
- Import required libraries and the LogisticRegression class.
- Load the Breast Cancer dataset using scikit-learn.
- Split the dataset into training and testing sets.
- Create an instance of the LogisticRegression class with desired parameters.
- Fit the model using the training data.
- Predict using the testing data and calculate accuracy.