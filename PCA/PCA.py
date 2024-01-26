import numpy as np

class PCA:
    def __init__(self, n_components=2):
        """
        Principal Component Analysis (PCA) constructor.

        Parameters:
        - n_components (int): Number of principal components to retain (default is 2).
        """
        self.n_components = n_components
        self.components = None  # Principal components
        self.feature_mean = None  # Mean of each feature

    def fit(self, X):
        """
        Fit the PCA model to the input data.

        Parameters:
        - X (numpy.ndarray): Input data matrix with rows as samples and columns as features.

        Notes:
        - Updates self.feature_mean with mean values of each feature.
        - Computes and retains the top self.n_components principal components in self.components.
        """
        # Calculate means of all features and shift the data to center around the origin by subtracting means
        self.feature_mean = np.mean(X, axis=0)
        X = X - self.feature_mean

        # Calculate covariance matrix of features (dimension n_feature x n_feature)
        feature_covariance = np.cov(X.T)  # Transpose to have row-wise features

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigen_vectors, eigen_values = np.linalg.eig(feature_covariance)

        # Transpose eigen_vectors back to row-sample and column-features form
        # After transformation, rows are original features, and columns are corresponding principal components (eigenvectors).
        # Values are weights assigned to each feature for the corresponding principal components.
        eigen_vectors = eigen_vectors.T

        # Sort eigen_vectors in descending order based on corresponding eigen_values
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[idx]

        # Select top n_components from eigen_vectors as principal components
        self.components = eigen_vectors[:self.n_components]

    def transform(self, X):
        """
        Transform input data to the reduced-dimensional space defined by the principal components.

        Parameters:
        - X (numpy.ndarray): Input data matrix with rows as samples and columns as features.

        Returns:
        - numpy.ndarray: Transformed data matrix in the reduced-dimensional space.
        """
        X = X - self.feature_mean
        return np.dot(X, self.components.T)
