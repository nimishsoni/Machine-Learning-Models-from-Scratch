import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from NaiveBayes import GaussianNaiveBayes

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters:
    - y_true (numpy.ndarray): True class labels.
    - y_pred (numpy.ndarray): Predicted class labels.

    Returns:
    - accuracy (float): Classification accuracy.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Generate synthetic dataset
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize and fit the NaiveBayes model
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)

# Make predictions on the test set
predictions = nb.predict(X_test)

# Evaluate and print the classification accuracy
print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))

# Plot Probability Distribution Function of features.
nb.plot_pdf(0)
nb.plot_pdf(1)
