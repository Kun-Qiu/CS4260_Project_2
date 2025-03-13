import numpy as np

class TreeNode:
    """
    ==============================
    TreeNode Class
    ==============================
    """
    def __init__(self, attribute=None, threshold=None, left_child=None, right_child=None, value=None): 
        self.attribute      = attribute
        self.threshold      = threshold
        self.left_child     = left_child
        self.right_child    = right_child
        self.value          = value


class DecisionTree:
    """
    ==============================
    Decision Tree Class
    ==============================
    """
    def __init__(self, data, attributes, label, depth=0, max_depth=3, min_samples_split=5):
        self.attributes     = attributes
        self.data           = data
        self.label          = label
        self.depth          = depth
        self.max_depth      = max_depth
        self.min_samples    = min_samples_split
        self.tree           = None

    @staticmethod
    def _entropy(y):
        """
        Calculate entropy of target labels
        """
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add epsilon to prevent log(0)

    def _information_gain(self, parent_y, left_y, right_y):
        """Calculate information gain from splitting"""
        parent_entropy = self._entropy(parent_y)
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)
        child_entropy = (left_weight * self._entropy(left_y) + 
                         right_weight * self._entropy(right_y))
        return parent_entropy - child_entropy

    def _split(self, data, threshold):
        """Split data into left and right based on attribute threshold"""
        left_indices    = np.where(data <= threshold)
        right_indices   = np.where(data > threshold)
        return left_indices, right_indices

    def _find_best_split(self, data):
        """
        ===============================================
        Find best attribute and threshold for splitting
        ===============================================
        """
        best_gain = -float('inf')
        best_attr, best_thresh = None, None
        
        label = np.array(self.label)

        for attribute in self.attributes:
            values = np.array(data[attribute])
            for threshold in values:
                left_indices, right_indices = self._split(values, threshold)
                if not left_indices or not right_indices:
                    continue
                left, right = values[left_indices], values[right_indices]
                left_y, right_y = label[left_indices], label[right_indices]
                gain = self._information_gain(self.label, left_y, right_y)
                print(gain)

                if gain > best_gain:
                    best_gain = gain
                    best_attr = attribute
                    best_thresh = threshold

        return best_attr, best_thresh, best_gain

    def build_tree(self):
        """
        ================================
        Recursively build decision tree
        ================================
        """
        # Stopping conditions
        if (self.depth >= self.max_depth or 
            len(self.data) < self.min_samples or 
            len(set(self.label)) == 1):
            counts = np.bincount(self.label)
            return TreeNode(value=np.argmax(counts))

        # Find best split
        best_attr, best_thresh, best_gain = self._find_best_split(self.data)
        if best_gain <= 0:
            counts = np.bincount(self.label)
            return TreeNode(value=np.argmax(counts))

        # Split data and build subtrees
        left_data, right_data = self._split(self.data, best_attr, best_thresh)
        left_child = DecisionTree(
            left_data, self.attributes, self.label, 
            self.depth + 1, self.max_depth, self.min_samples
        ).build_tree()
        right_child = DecisionTree(
            right_data, self.attributes, self.label,
            self.depth + 1, self.max_depth, self.min_samples
        ).build_tree()

        return TreeNode(
            attribute=best_attr,
            threshold=best_thresh,
            left_child=left_child,
            right_child=right_child
        )

    # def predict(self, row):
    #     """Predict class for a single data row"""
    #     node = self.tree
    #     while node.value is None:
    #         if row[node.attribute] <= node.threshold:
    #             node = node.left_child
    #         else:
    #             node = node.right_child
    #     return node.value