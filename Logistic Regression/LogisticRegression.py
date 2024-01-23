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

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the logistic regression model.

        Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
        val_set (tuple): Validation set as a tuple (X_val, y_val).
        """
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        best_val_loss = float('inf')
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

            # Compute training loss
            train_loss = self._compute_loss(X, y)
            self.training_loss.append(train_loss)

            # Early stopping
            if self.early_stopping and X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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
    
    def _compute_loss(self, X, y):
        """
        Compute the logistic loss for the given input features and labels.

        Parameters:
        - X (numpy.ndarray): Input features, where each row represents a sample and each column a feature.
        - y (numpy.ndarray): Actual labels corresponding to the input features.

        Returns:
        - float: Logistic loss calculated using cross-entropy for binary classification.

        Notes:
        - The loss is computed based on the sigmoid activation of linear predictions.
        - Cross-entropy loss is used for binary classification, comparing predicted probabilities to actual labels.
        - The loss is the negative mean of the sum of log probabilities for positive and negative classes.
        """

        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss


