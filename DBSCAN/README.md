# DBSCAN Clustering Algorithm

## Overview

This repository contains an implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. DBSCAN is a density-based clustering algorithm that groups together data points based on their proximity in the feature space. The implementation is provided in the `DBSCAN.py` file.

## Files

- **DBSCAN.py:** This file contains the implementation of the DBSCAN clustering algorithm. The class `DBSCAN` is defined with methods for initializing the algorithm, fitting the model to data, and performing clustering.

## Usage

To use the DBSCAN algorithm on your own dataset:

1. Copy the `DBSCAN.py` file into your project.
2. Import the `DBSCAN` class from `DBSCAN.py`.
3. Create an instance of the `DBSCAN` class and use its `predict` method to obtain cluster labels for your dataset.

### Example:

```python
from DBSCAN import DBSCAN
import numpy as np

# Create an instance of DBSCAN
dbscan_model = DBSCAN(eps=1, min_pts=5)

# Your dataset (replace with your own data)
X = np.array([[1, 2], [2, 3], [5, 6], [7, 8], [8, 9]])

# Apply DBSCAN
cluster_labels = dbscan_model.predict(X)

# Print the resulting cluster labels
print("Cluster Labels:", cluster_labels)
Parameters
eps: The maximum distance between two samples for them to be considered as in the same neighborhood. Default value is 1.

min_pts: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Default value is 5.

**Methods**
predict(X): Apply the DBSCAN algorithm to the input dataset X and return an array of cluster labels.

**Dependencies**
NumPy
Example
To see the DBSCAN algorithm in action, you can run the provided example script:

python apply_DBSCAN.py
License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgments
Inspired by the original DBSCAN paper by Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (Link to Paper)
Feel free to contribute, report issues, or provide feedback. Happy clustering!
