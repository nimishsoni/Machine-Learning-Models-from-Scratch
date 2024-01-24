# KMeans Clustering

This repository contains Python code for KMeans clustering algorithm along with an example script to apply the KMeans algorithm on synthetic data.

## Files

1. **KMeans.py**: This file contains the implementation of the KMeans class. The class provides methods for fitting the model to data, making predictions, and visualizing the clustering results.

2. **apply_KMeans.py**: An example script demonstrating how to use the KMeans class on synthetic data. It generates a synthetic dataset with three clusters using the `make_blobs` function from scikit-learn, initializes a KMeans model, fits the model to the data, predicts cluster labels, and visualizes the results.

3. **KMeans++.py**: This file inherits KMeans class implementation and modifies the initialization method to probabilistic proportional to distance between existing closest centroid and sample data points. This initialization strategy ensures that centroids initialized are far away from each other.

## Usage

To use the KMeans algorithm on your own data, follow these steps:

1. Import the `KMeans` class from the `KMeans.py` file. (for KMeans++ replace the name of the file as KMeans++.py)
2. Create an instance of the `KMeans` class with the desired number of clusters (`K`) and other parameters.
3. Use the `fit` method to fit the model to your data.
4. Use the `predict` method to assign cluster labels to your data.
5. Optionally, use the `plot` method to visualize the clustering results.
