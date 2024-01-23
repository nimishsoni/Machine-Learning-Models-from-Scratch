import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - x1 (numpy array): First point.
    - x2 (numpy array): Second point.

    Returns:
    - float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=4, n_iters=100, plot_steps=False):
        """
        Initialize KMeans clustering model.

        Parameters:
        - K (int): Number of clusters.
        - n_iters (int): Number of iterations for the KMeans algorithm.
        - plot_steps (bool): Flag to enable/disable plotting of intermediate steps.
        """
        self.K = K
        self.n_iters = n_iters
        self.plot_steps = plot_steps
        # Initialize an empty list for each cluster to hold data point indices
        self.clusters = [[] for _ in range(self.K)]
        # List of cluster centroids
        self.centroids = []

    def predict(self, X):
        """
        Perform KMeans clustering on input data.

        Parameters:
        - X (numpy array): Input data.

        Returns:
        - numpy array: Predicted cluster labels for each data point.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize the centroids
        random_centroid_ids = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[id] for id in random_centroid_ids]

        # Clustering step
        for _ in range(self.n_iters):

            # Assign each sample to the closest centroid
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Update centroids based on the mean of existing cluster sample list
            old_centroids = self.centroids
            self.centroids = [np.mean(self.X[cluster], axis=0) for cluster in self.clusters]

            # Assign cluster labels to all the samples
            labels = np.empty(self.n_samples)
            for cluster_idx, cluster in enumerate(self.clusters):
                for sample_idx in cluster:
                    labels[sample_idx] = cluster_idx

            # Check if the clusters have converged
            distances = [euclidean_distance(self.centroids[i], old_centroids[i]) for i in range(self.K)]

            if sum(distances) == 0:
                break

        return labels

    def _create_clusters(self, centroids):
        """
        Assign each sample to the closest centroids.

        Parameters:
        - centroids (list): List of cluster centroids.

        Returns:
        - list: List of clusters, where each cluster contains indices of samples.
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Find the index of the closest centroid to a given sample.

        Parameters:
        - sample (numpy array): Input data point.
        - centroids (list): List of cluster centroids.

        Returns:
        - int: Index of the closest centroid.
        """
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def plot(self):
        """
        Plot the current state of the KMeans clustering.

        This method visualizes the data points and centroids.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()