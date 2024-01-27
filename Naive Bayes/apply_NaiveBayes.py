import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from NaiveBayes import GaussianNaiveBayes, MultinomialNaiveBayes

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

# Initialize and fit the Gaussian NaiveBayes model
gaussian_nb = GaussianNaiveBayes()
gaussian_nb.fit(X_train, y_train)

# Make predictions on the test set using Gaussian NaiveBayes
gaussian_predictions = gaussian_nb.predict(X_test)

# Evaluate and print the classification accuracy for Gaussian NaiveBayes
print("Gaussian Naive Bayes classification accuracy:", accuracy(y_test, gaussian_predictions))

# Plot Probability Distribution Function of features for Gaussian NaiveBayes
gaussian_nb.plot_pdf(0)
gaussian_nb.plot_pdf(1)

# Initialize and fit the Multinomial NaiveBayes model
multinomial_nb = MultinomialNaiveBayes()
multinomial_nb.fit(X_train, y_train)

# Make predictions on the test set using Multinomial NaiveBayes
multinomial_predictions = multinomial_nb.predict(X_test)

# Evaluate and print the classification accuracy for Multinomial NaiveBayes
print("Multinomial Naive Bayes classification accuracy:", accuracy(y_test, multinomial_predictions))