**Decision Tree**
**Files**:

**DecisionTree.py**:
This file contains the implementation of the Decision Tree model from scratch.
Two main classes are defined:
Node: Responsible for creating and initializing internal and leaf nodes with respective features.
Decision Tree: Responsible for building the decision tree.

**applyDT.py**:
This file leverages the Decision Tree model to train and predict outcomes.

**Key Steps in DecisionTree Class**:

**Input Parameters**:
Get input parameters for growing a decision tree, such as the number of features to be used, maximum depth of the tree, and the minimum number of samples required for a node to be considered for splitting.

**Tree Growth**:
Considering the number of features and tree constraints, grow trees by splitting nodes into two. The feature and threshold are decided based on the maximum information gain.

**Leaf Node Creation**:

The trees are grown up to the leaf nodes, which are not split further either due to constraints or when the remaining samples are from a single class. The outcome from this node is computed as the most common label from the remaining samples.

**Model Prediction**:
Once the model training (growing necessary trees) is completed, it can be used for predicting output using the applyDT.py file.
Feel free to explore, contribute, or use these implementations to gain a deeper understanding of machine learning models. If you have any questions or suggestions, please let me know!
  
