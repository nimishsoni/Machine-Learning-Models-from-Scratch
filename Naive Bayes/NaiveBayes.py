import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
        - X (numpy.ndarray): 2D array representing input data.
        - y (numpy.ndarray): 1D array representing target labels.
        """
        n_samples, n_features = X.shape  # Get the number of samples and features
        self._classes = np.unique(y)  # Assign unique values in y to classes
        n_classes = len(self._classes)

        # Initialize arrays for mean, variance, and prior probability for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priorprob = np.zeros(n_classes, dtype=np.float64)

        # Compute mean, variance, and prior probability for each class
        for idx, cl in enumerate(self._classes):
            X_cl = X[y == cl] 

            # Compute mean and variance for each feature in the current class
            self._mean[idx, :] = X_cl.mean(axis=0)
            self._var[idx, :] = X_cl.var(axis=0)

            # Calculate prior probability based on the frequency of occurrence of the class
            self._priorprob[idx] = X_cl.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X (numpy.ndarray): 2D array representing input data.

        Returns:
        - y_pred (list): Predicted class labels for each input sample.
        """
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        """
        Predict the class label for a single input sample.

        Parameters:
        - x (numpy.ndarray): Input sample.

        Returns:
        - predicted_class: Predicted class label for the input sample.
        """
        posterior_prob_list = []

        # Calculate posterior probability for each class
        for idx, cl in enumerate(self._classes):
            prior_prob = np.log(self._priorprob[idx])
            posterior_prob = np.sum(np.log(self._pdf(idx, x)))
            posterior_prob = posterior_prob + prior_prob
            posterior_prob_list.append(posterior_prob)

        return self._classes[np.argmax(posterior_prob_list)]
    
    def _pdf(self, idx, x):
        """
        Calculate the probability density function for a feature in a given class.

        Parameters:
        - idx (int): Index representing the class.
        - x (numpy.ndarray): Input sample.

        Returns:
        - pdf (numpy.ndarray): Probability density function for the input sample.
        """
        mean_cl = self._mean[idx]
        var_cl = self._var[idx]
        numerator = np.exp(-(x - mean_cl) ** 2 / (2 * var_cl))
        denominator = np.sqrt(2 * np.pi * var_cl)
        return numerator / denominator