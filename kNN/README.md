**kNN Classifier with Standardization**
- This repository contains Python code implementing a k-nearest neighbors (kNN) classifier with the ability to standardize features. The kNN algorithm is a simple and effective classification algorithm based on the majority vote of its k-nearest neighbors.

**Files**
kNN_modified.py: Python script containing the implementation of the modified kNN classifier.
apply_kNN.py: Example script demonstrating the usage of the modified kNN classifier on the Iris dataset.

**Usage**
kNN_modified.py
The kNN_modified.py script defines a kNN class with the following features:

- Standardization: The ability to standardize input features for improved performance.
- Predictions: Making predictions based on the k-nearest neighbors.
- Initialization: Initializing the kNN classifier with the desired value of k.

**apply_kNN.py**
The apply_kNN.py script demonstrates how to use the modified kNN classifier:

- Load the Iris Dataset: Load the Iris dataset from scikit-learn's built-in datasets.
- Split the Dataset: Split the dataset into training and testing sets.
- Visualize the Dataset: Visualize the dataset using a scatter plot.
- Create and Fit the kNN Model: Create a kNN model with k=5 and standardize features. Fit the model with the training data.
- Make Predictions: Use the trained model to make predictions on the test set.
- Print Predictions and Accuracy: Print the predictions and calculate the accuracy of the model.

**Dependencies**
NumPy
scikit-learn
Matplotlib

**How to Run**
Ensure you have the required dependencies installed (pip install numpy scikit-learn matplotlib).
Run the apply_kNN.py script (python apply_kNN.py).

Contributors
Nimish Soni

Feel free to contribute or report issues!

