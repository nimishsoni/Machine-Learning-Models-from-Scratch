import numpy as np

class LinearRegression:
    """ Models the linear relationship between indeipendent features and dependent variable
    """
    def __init__(self, lr=0.001, n_iters = 1000):
        """ Method to Initialize the class parameters"""
        self.learning_rate = lr 
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """This method is responsible for training the linear regression model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters): # Run training loop for n_iters
            y_pred = np.dot(X, self.weights) + self.bias #Predict output using current weights and bias

            # Update the weights and bias using computed gradients and learning rate
            # Gradient computation by taking derivative of mean square error loss function with respect to 
            dW = 1/n_samples * np.dot(X.T, (y_pred - y))
            dB = 1/n_samples * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * dB

        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

