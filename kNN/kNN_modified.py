import numpy as np
from collections import Counter
        
class kNN:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform', standardize_features=True):
        """
        Initialize kNN classifier.

        Parameters:
        - k (int): Number of neighbors to consider.
        - distance_metric (str): Distance metric for kNN, e.g., 'euclidean', 'manhattan'.
        - weights (str): Weighting strategy for neighbors, 'uniform' or 'distance'.
        - standardize_features (bool): Whether to standardize input features.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.standardize_features = standardize_features
        
    def fit(self, X, y):
        """
        Fit the kNN classifier with training data.

        Parameters:
        - X (numpy array): Training data features.
        - y (numpy array): Training data labels.
        """
        if self.standardize_features:
            self.mean, self.std = self._calculate_mean_and_std(X)
            X = self._standardize_features(X)
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
        if self.standardize_features:
            X = self._standardize_features(X)
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
        # Compute the distance
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]

        # Get the closest k point labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote with tie-breaking
        counter = Counter(k_nearest_labels)
        most_common = counter.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Handle tie-breaking by choosing the label with the smallest distance
            distances_of_tied_labels = [distances[i] for i in k_indices if self.y_train[i] == most_common[0][0]]
            closest_tied_label_index = np.argmin(distances_of_tied_labels)
            return self.y_train[k_indices[closest_tied_label_index]]
        else:
            return most_common[0][0]

    def _calculate_distance(self, x1, x2):
        """
        Calculate the distance between two data points.

        Parameters:
        - x1 (numpy array): First data point.
        - x2 (numpy array): Second data point.

        Returns:
        - float: Distance between x1 and x2.
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Invalid distance metric. Supported metrics: 'euclidean', 'manhattan'")

    def _calculate_mean_and_std(self, X):
        """
        Calculate mean and standard deviation for standardization.

        Parameters:
        - X (numpy array): Input data features.

        Returns:
        - tuple: Mean and standard deviation.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return mean, std

    def _standardize_features(self, X):
        """
        Standardize input features.

        Parameters:
        - X (numpy array): Input data features.

        Returns:
        - numpy array: Standardized features.
        """
        return (X - self.mean) / self.std
