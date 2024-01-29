# The code implements Decision Tree from scratch

import numpy as np
from collections import Counter
# Define Node Class representing internal and Leaf nodes
# feature, threshold, 
class Node:
    def __init__(self, feature = None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature# feature used for split, none for leaf node
        self.threshold = threshold # Threhold of feature used for split, none for leaf node
        self.left = left # left child node, none for leaf node
        self.right = right #right child none, none for leaf node
        self.value=None # Predicted output or class for leaf node, none for internal node

    def is_leaf_node(self):
        return self.value is not None #Returns True if value is non None in case of leaf node

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth = 100, n_features=None):
        self.min_samples_split = min_samples_split # minimium samples to be considered for an internal node split
        self.max_depth = max_depth # maximum depth of the decision tree to control the complexity and hence overfitting
        self.n_features = n_features # Number of features (subset) to be considered for growth of tree
        self.root = None
    
    def fit(self, X, y): # responsible for fitting the decision tree to the training data.
        # sets the number of features to consider for splitting. If self.n_features is not specified, it takes all features.
        # Otherwise, it considers the minimum of the total features and self.n_features.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        
        # Calls the _grow_tree method to initiate the growth of the decision tree.
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth = 0): #  recursive method responsible for growing the decision tree.
        
        n_samples, n_feats = X.shape # number of samples (rows) and features in the current node
        n_labels = len(np.unique(y)) #represents the number of unique labels in the current node.

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y) # compute prediction for the leaf
            return Node(value=leaf_value)
        
            # check if best_thresh is None

        # Randomly selects a subset of features (self.n_features)  without replacement.
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # find the best split - Best feature and Best threshold
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)


        # create child nodes
        # Splits the data into left and right indices based on the best feature and threshold.
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        
        # Recursively grows the left and right child node.
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        # Initializes a variable to keep track of the best information gain.
        best_gain = -1
        # Initialize variables for best feature index and split threshold
        split_idx, split_threshold = None, None

        # Iterate over all the features from randomly selected feature ids
        # to compute best feature and threshold for splitting current internal node
        for feat_idx in feat_idxs:
            #select the column from node data with the feature id
            X_column = X[:, feat_idx]
            # Identify set of unique possible thresholds from the selected column
            thresholds = np.unique(X_column)

            # Iterate overall all thresholds to compute inforrmation gain for each and select the Best one

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold): # Calculates information gain for potential split based on feature
        # parent entropy
        parent_entropy = self._entropy(y) 

        # create children using feature column data from node based on threshold
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y) # Length of parent node
        n_l, n_r = len(left_idxs), len(right_idxs) # number of data sample in left and right child nodes
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) # Entropy of left and right child nodes
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r # Weighted average of entropy based on number of samples in child node
        
        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y) # Generates a histogram of class labels.
        ps = hist / len(y) # Compute probability of each label
        return -np.sum([p * np.log(p) for p in ps if p>0]) #Compute entropy


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X): 
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
