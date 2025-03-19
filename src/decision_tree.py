import numpy as np

class TreeNode:
    """
    ================
    TreeNode Class
    ================
    """
    def __init__(self, attribute_idx=None, threshold=None, left_child=None, right_child=None, value=None): 
        self.attribute_idx  = attribute_idx
        self.threshold      = threshold
        self.left_child     = left_child
        self.right_child    = right_child
        self.value          = value


class DecisionTree:
    def __init__(self, X, y, depth=0, max_depth=3, num_thresh=10):
        """
        Default constructor
        :params X           :   Dataset (shape : [samples, features])
        :params y           :   Label array (shape: [samples, 1])
        :params depth       :   Current depth of the tree
        :params max_depth   :   Max depth of the tree
        """
        self.X           = X
        self.y           = y
        self.depth       = depth
        self.max_depth   = max_depth
        self.tree        = None
        self.num_thresh  = num_thresh
    
    
    @staticmethod
    def __entropy(y):
        """
        Calculate the entropy of a label array y
        :params y   :   Label array (shape: [samples, 1])
        :returns    :   Entropy of label array
        """
        counts          = np.bincount(y)
        probabilities   = counts / len(y)
        eps             = np.finfo(float).eps 
        return np.sum(probabilities * np.abs(np.log2(probabilities + eps)))
    

    def __information_gain(self, parent_y, left_y, right_y):
        """
        Information Gain = Entropy(Parent) - Weighted Sum of Child Entropy
        :params parent_y : Label array for the parent
        :params left_y   : Label array for the left child
        :params right_y  : Label array for the right child
        :returns         : The information gain
        """
        
        parent_entropy = self.__entropy(parent_y)
        n       = len(parent_y)
        n_left  = len(left_y)
        n_right = len(right_y)
        
        if n == 0 or n_left == 0 or n_right == 0:
            # Edge cases 
            return 0  
            
        left_entropy  = self.__entropy(left_y)
        right_entropy = self.__entropy(right_y)
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        
        return parent_entropy - child_entropy
    
    
    def __split_dataset(self, X, y, attr_index, thresh):
        """
        Given an attribute index and threshold, splits the data into left and right partitions.
        :params X          : Feature dataset
        :params y          : Label data
        :params attr_index : Feature index 
        :params thresh     : Threshold
        :returns           : (X_left, y_left), (X_right, y_right).
        """
        left_mask  = X[:, attr_index] <= thresh
        right_mask = X[:, attr_index] > thresh
        
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
    
    
    def __find_best_split(self, X, y):
        """
        Finds the best attribute and threshold among the given attributes to split on,
        based on the highest information gain.
        :params X : Input data
        :params y : Labeled data
        :returns  : The optimal attribute_idx, optimal threshold and optimal entropy
        """
        opt_entropy     = np.inf
        opt_attr        = None
        opt_thresh      = None
        
        for idx in range(self.X.shape[1]):
            """
            Divide thresholds into equal poritons to avoid checking for every values
            in the attribute 
            """
            max_val, min_val = int(np.max(X[:, idx])), int(np.min(X[:, idx]))
            thresholds = np.linspace(min_val, max_val, self.num_thresh)
            
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = self.__split_dataset(X, y, idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    # Skip splits that result in empty sets
                    continue
                entropy = (len(X_left) / len(X) * self.__entropy(y_left) + 
                           len(X_right) / len(X) * self.__entropy(y_right))
                
                if entropy < opt_entropy:
                    opt_entropy   = entropy
                    opt_attr      = idx
                    opt_thresh    = threshold
        
        return opt_attr, opt_thresh, opt_entropy
    
    
    def __majority_class(self, y):
        """
        Returns the most common class in y.
        :params y : Label array (shape: [samples])
        :returns  : The majority class in labeled data
        """
        return np.argmax(np.bincount(y))
    
    
    def __build_tree_recursive(self, X, y, depth):
        """
        Building the decision tree recursively. Build the tree according to
        the stopping criteria.
        :params X       : Input data
        :params y       : Label data
        :params depth   : Depth
        :returns        : None
        """

        if depth >= self.max_depth:
            # Stopping criteria 1: max depth
            return TreeNode(value=self.__majority_class(y))
                
        if len(np.unique(y)) == 1:
            # Stopping criteria 2: all labels are the same
            return TreeNode(value=y[0])
        
        best_attr, best_threshold, _ = self.__find_best_split(X, y)
        (X_left, y_left), (X_right, y_right) = self.__split_dataset(X, y, best_attr, best_threshold)

        # Stopping criteria 3 : Information Gain
        if best_attr is None or self.__information_gain(y, y_left, y_right) <= 1e-3:
            return TreeNode(value=self.__majority_class(y))
        
        left_child  = self.__build_tree_recursive(X_left, y_left, depth + 1)
        right_child = self.__build_tree_recursive(X_right, y_right, depth + 1)
        
        return TreeNode(attribute_idx=best_attr,
                        threshold=best_threshold,
                        left_child=left_child,
                        right_child=right_child)
    
    
    def build_tree(self):
        """
        Recursively builds the decision tree and saves root in self.tree.
        """
        self.tree = self.__build_tree_recursive(self.X, self.y, self.depth)

    
    def predict(self, X):
        """
        Predicts the label for a single data row. Assumes self.tree is already built.
        :params X : Input data
        :returns  : Prediction
        """
        
        assert self.tree is not None, "The decision tree has not been built yet. Call build_tree() first."

        node = self.tree
        while node.value is None:  # Traverse until leaf node
            if X[node.attribute_idx] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.value
    
    
    def predict_batch(self, X_test):
        """
        Predicts labels for a batch of rows in X_test.
        """
        predictions = [self.predict(x) for x in X_test]
        return np.array(predictions)