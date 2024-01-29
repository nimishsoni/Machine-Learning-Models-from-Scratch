import numpy as np
from DecisionTree import DecisionTree
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        Initialize the Random Forest Classifier.

        Parameters:
        - n_trees: Number of trees in the forest.
        - max_depth: Maximum depth of each decision tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - n_features: Number of features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest model on the given training data.

        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        # Grow n_trees each of max_depth
        for iteration in range(self.n_trees):
            # Initialize decision tree with given parameters
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            # Bootstrap samples
            X_sample, y_sample = self._bootstrap_samples(X, y)
            # Fit the decision tree on the bootstrapped samples
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Generate bootstrapped samples for training the decision trees.

        Parameters:
        - X: Features.
        - y: Labels.

        Returns:
        - X_sample, y_sample: Bootstrapped samples.
        """
        # Randomly select 50% of the samples
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples // 2, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Determine the most common label in an array.

        Parameters:
        - y: Array of labels.

        Returns:
        - most_common: The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Make predictions using the Random Forest model.

        Parameters:
        - X: Features for prediction.

        Returns:
        - predictions: Array of predicted labels.
        """
        y_pred = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(y_pred, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions