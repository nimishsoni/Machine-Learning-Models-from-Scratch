import numpy as np

class LinearRegression:
    """ Models the linear relationship between indeipendent features and dependent variable
    """
    def __init__(self, lr=0.001, n_iters = 1000, regularization = None, alpha = 0.01, decay_rate = 0.0001, early_stopping = False, patience = 10):
        """ Method to Initialize the class parameters"""
        self.learning_rate = lr 
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.training_errors = []
        self.early_stopping = early_stopping
        self.patience = patience

    def fit(self, X, y, val_set = None):
        """This method is responsible for training the linear regression model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        best_weights = np.copy(self.weights)
        best_bias = self.bias
        best_val_loss = float('inf')
        patience_count = 0

        for iteration_number in range(self.n_iters): # Run training loop for n_iters
            y_pred = np.dot(X, self.weights) + self.bias #Predict output using current weights and bias

            # L1 and L2 Regularization
            if self.regularization == 'l2':
                reg_term = self.alpha * np.sum(self.weights**2)
            
            elif self.regularization == 'l1':
                reg_term = self.alpha * np.sum(np.abs(self.weights))
            
            else:
                reg_term = 0
            
            mse_loss = (y_pred - y)**2 + reg_term

            # Implement learning rate schedule to iteratively adjust learning rate during training for better convergenece
            self.learning_rate = self.learning_rate / (1 + self.decay_rate * iteration_number)


            # Update the weights and bias using computed gradients and learning rate
            # Gradient computation by taking derivative of mean square error loss function with respect to 
            dW = 1/n_samples * (np.dot(X.T, (y_pred - y)) + reg_term * np.sign(self.weights))
            dB = 1/n_samples * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * dB
           
            self.training_errors.append(mse_loss)

            # Early stopping and Saving Best model parameters
            if self.early_stopping and val_set is not None:
                #print(val_set[0].shape, val_set[1].shape)
                val_loss = np.mean((np.dot(val_set[0], self.weights) + self.bias)-val_set[1])**2 + reg_term
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = np.copy(self.weights)
                    best_bias = self.bias
                    patience_count = 0
                else:
                    patience_count+=1
                    if patience_count >=self.patience:
                        break

            # Use the best weights and bias
            self.weights = best_weights
            self.bias = best_bias
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


