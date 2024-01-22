import numpy as np

def sigmoid(x):
    """Sigmoid activation function.

    Args:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    """Logistic Regression classifier with options for regularization and early stopping.

    Args:
    lr (float): Learning rate for gradient descent.
    decay_rate (float): Rate at which the learning rate decays.
    n_iters (int): Number of iterations for training.
    regularization (str): Type of regularization, 'l1', 'l2', or None.
    alpha (float): Regularization strength.
    early_stopping (bool): Enable early stopping.
    patience (int): Number of iterations to wait for improvement during early stopping.
    """

    def __init__(self, lr=0.0001, decay_rate=0.01, n_iters=1000, regularization=None, alpha=0.01, early_stopping=False, patience=10):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.decay_rate = 0.01
        self.n_iters = n_iters
        self.regularization = regularization
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.patience = patience
        self.training_loss = []

    def fit(self, X, y, val_set=None):
        """Train the logistic regression model.

        Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
        val_set (tuple): Validation set as a tuple (X_val, y_val).
        """
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        best_accuracy = 0.0
        best_weights = np.copy(self.weights)
        best_bias = self.bias
        patience_count = 0

        for iteration in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)

            if self.regularization == 'l1':
                reg_term = self.alpha * np.sum(np.abs(self.weights))
            elif self.regularization == 'l2':
                reg_term = self.alpha * np.sum(self.weights ** 2)
            else:
                reg_term = 0
            
            # Implement learning rate schedule
            self.lr = self.lr / (1 + self.decay_rate * iteration)

            dW = 1/n_sample * np.dot(X.T, (y_pred - y))
            dB = 1/n_sample * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * (dW + reg_term)
            self.bias = self.bias - self.lr * dB

            if self.early_stopping and val_set is not None:
                val_pred = sigmoid(np.dot(val_set[0], self.weights) + self.bias)
                val_class = [0 if y < 0.5 else 1 for y in val_pred]
                val_accuracy = np.sum(val_class == val_set[1]) / len(val_set[1])
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_weights = np.copy(self.weights)
                    best_bias = self.bias
                    patience_count = 0
                
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
            
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Make predictions using the trained model.

        Args:
        X (numpy.ndarray): Input features.

        Returns:
        list: Predicted class labels.
        """
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        y_class = [0 if y < 0.5 else 1 for y in y_pred]
        return y_class


