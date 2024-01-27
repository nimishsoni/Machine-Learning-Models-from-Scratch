# This code implements Gaussian Naive Bayes Algorithm
import numpy as np
import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.
        Fitting here means computing class-wise Gaussian probability distribution function through mean and square for each feature.
        This probability distribution along with prior class probability will be used to then estimate class for new input sample having feature values

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
        - x (numpy.ndarray): Input sample containing features.

        Returns:
        - predicted_class: Predicted class label for the input sample.
        """
        posterior_prob_list = []

        # Calculate posterior probability for each class
        # Likelihood of x = input value given probability distribution for the class having mean and variance as calculated before
        for idx, cl in enumerate(self._classes):
            prior_prob = np.log(self._priorprob[idx]) 
            posterior_prob = np.sum(np.log(self._pdf(idx, x))) # Posterior probability is calculated for each class based on the log likelihoods of the features in the input sample
            posterior_prob = posterior_prob + prior_prob 
            posterior_prob_list.append(posterior_prob) 

        return self._classes[np.argmax(posterior_prob_list)]
    
    def _pdf(self, idx, x):
        """
        Calculate the Gaussian probability density function for each feature for a given class. 
        It is the probability of 

        Parameters:
        - idx (int): Index representing the class.
        - x (numpy.ndarray): Input sample.

        Returns:
        - pdf (numpy.ndarray): Probability density function for the input sample.
        """
        mean_cl = self._mean[idx]
        var_cl = self._var[idx]
        numerator = np.exp(-(x - mean_cl) ** 2 / (2 * var_cl)) #1-D Array of size equal to n_features
        denominator = np.sqrt(2 * np.pi * var_cl) # Normalize the distribution
        return numerator / denominator

    def plot_pdf(self, feature_index):
        """
        Plot the Gaussian Probability Distribution Functions for each class for a specific feature.

        Parameters:
        - feature_index (int): Index of the feature to plot.
        """
        if feature_index < 0 or feature_index >= self._mean.shape[1]:
            raise ValueError("Invalid feature index")

        plt.figure(figsize=(10, 6))
        for idx, cl in enumerate(self._classes):
            mean_cl = self._mean[idx, feature_index]
            var_cl = self._var[idx, feature_index]
            x_values = np.linspace(mean_cl - 3 * np.sqrt(var_cl), mean_cl + 3 * np.sqrt(var_cl), 100)
            pdf_values = np.exp(-(x_values - mean_cl) ** 2 / (2 * var_cl)) / np.sqrt(2 * np.pi * var_cl)
            plt.plot(x_values, pdf_values, label=f'Class {cl}')

        plt.title(f'Class-wise Gaussian PDF for Feature {feature_index}')
        plt.xlabel(f'Feature {feature_index} values')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

class MultinomialNaiveBayes(GaussianNaiveBayes):
    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes model to the training data.

        Parameters:
        - X (numpy.ndarray): 2D array representing input data.
        - y (numpy.ndarray): 1D array representing target labels.
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize arrays for class-wise feature counts and class priors
        self._class_feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        self._class_counts = np.zeros(n_classes, dtype=np.float64)

        # Update feature counts and class counts based on training data
        for idx, cl in enumerate(self._classes):
            X_cl = X[y == cl]
            self._class_feature_counts[idx, :] = np.sum(X_cl, axis=0)
            self._class_counts[idx] = X_cl.shape[0]

    def _predict(self, x):
        """
        Predict the class label for a single input sample using Multinomial Naive Bayes.

        Parameters:
        - x (numpy.ndarray): Input sample.

        Returns:
        - predicted_class: Predicted class label for the input sample.
        """
        log_likelihoods = np.zeros(len(self._classes))

        for idx, cl in enumerate(self._classes):
            log_prior = np.log(self._class_counts[idx] / np.sum(self._class_counts))
            log_likelihood = np.sum(x * np.log((self._class_feature_counts[idx] + 1) / (np.sum(self._class_feature_counts[idx]) + len(self._class_feature_counts[idx]))))
            log_likelihoods[idx] = log_likelihood + log_prior

        return self._classes[np.argmax(log_likelihoods)]