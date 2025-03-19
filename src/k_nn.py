import numpy as np

"""
==============================
K Nearest Neighbors Classifier
==============================
"""
class KNearestNeighbors:
    def __init__(self, data, label, k=3, distance_metric='euclidean'):
        """
        Default constructor for initialization of KNN classifier
        
        :params data            : Training data (shape: [n_samples, n_features])
        :params labels          : Training labels (shape: [n_samples])
        :params k               : Number of neighbors for KNN
        :params distance_metric : Distance metric for KNN ('euclidean' or 'manhattan')
        """
        self.data               = np.array(data)
        self.label              = np.array(label)
        self.k                  = int(k)
        self.distance_metric    = distance_metric.lower()
        
    def _compute_distance(self, X):
        """
        Compute distance between two input samples

        :params X : Input sample (shape: [n_features])
        :returns  : Distance between X and training data (shape: [n_samples])
        """
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.data - X) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.data - X), axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return distances

    def predict(self, X):
        """
        Predict labels for test samples
        
        :params X : Input sample (shape: [n_features])
        :returns  : Predicted labels (binary classification)
        """
        X = np.array(X)

        distances   = self._compute_distance(X)
        k_indices   = np.argsort(distances)[:self.k]
        k_labels    = self.label[k_indices]
        
        return np.bincount(k_labels).argmax()