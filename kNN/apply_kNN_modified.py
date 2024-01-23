import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kNN_modified import kNN  # Assuming the modified kNN class is defined in a file named kNN_modified.py

# Define a colormap for plotting
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Visualize the dataset
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# Create a kNN model with k=5 and use standardized features
model = kNN(k=5, standardize_features=True)

# Fit the model with training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Print the predictions
print(predictions)

# Calculate and print the accuracy
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
