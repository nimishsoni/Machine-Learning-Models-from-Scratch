import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from KMeans import KMeans

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data with 3 clusters
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)

# Display the shape of the generated data
print("Shape of the generated data:", X.shape)

# Determine the number of true clusters in the data
true_clusters = len(np.unique(y))
print("Number of true clusters:", true_clusters)

# Initialize KMeans model with the correct number of clusters and other parameters
kmeans_model = KMeans(K=true_clusters, n_iters=150, plot_steps=True)

# Predict cluster labels using the KMeans algorithm
predicted_labels = kmeans_model.predict(X)

# Plot the final clustering results
kmeans_model.plot()
