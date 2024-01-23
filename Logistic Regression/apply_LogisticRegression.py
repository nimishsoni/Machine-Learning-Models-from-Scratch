import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Load Breast Cancer dataset from sklearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Logistic Regression model
model = LogisticRegression(lr=0.01, regularization='l2', alpha=0.001, decay_rate=0.001, early_stopping=True, patience=25)

# Train the Logistic Regression model on the training set and validate on the test set
model.fit(X_train, y_train, X_val = X_test, y_val = y_test)

# Make predictions on the test set using the trained model
y_pred = model.predict(X_test)

# Define a function to calculate accuracy
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

# Calculate and print the accuracy of the model on the test set
acc = accuracy(y_pred, y_test)
print(f"Accuracy: {acc}")
