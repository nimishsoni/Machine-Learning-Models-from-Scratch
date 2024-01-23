
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - x1 (numpy array): First point.
    - x2 (numpy array): Second point.

    Returns:
    - float: Euclidean distance between x1 and x2.
    """
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

class kNN:
    def __init__(self, k=3):
        """
        Initialize kNN classifier.

        Parameters:
        - k (int): Number of neighbors to consider.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the kNN classifier with training data.

        Parameters:
        - X (numpy array): Training data features.
        - y (numpy array): Training data labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for input data using kNN algorithm.

        Parameters:
        - X (numpy array): Input data features.

        Returns:
        - list: Predicted labels for each input data point.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """
        Predict label for a single data point using kNN algorithm.

        Parameters:
        - x (numpy array): Input data point.

        Returns:
        - int: Predicted label.
        """
        # Computes the Euclidean distance between the input data point x and all training data points in self.X_train
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] 

        # Get the closest k point labels
        k_indices = np.argsort(distances)[:self.k] #argsort returns indices of points with k least distances
        k_nearest_labels = [self.y_train[i] for i in k_indices] # Get labels corresponding to points with k least indices

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
