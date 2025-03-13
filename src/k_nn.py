import numpy as np
from sklearn.neighbors import KNeighborsClassifier

"""
============================
Five Fold Cross Validation
============================
"""
def cross_validate_knn(data, labels, k=3, distance_metric='euclidean', n_folds=5, sklearn_check=False):
    """
    Perform 5-fold cross-validation for KNearestNeighbors classifier 
    Report mean and std of accuracy.
    
    :params data            : Training data (shape: [n_samples, n_features])
    :params labels          : Training labels (shape: [n_samples])
    :params k               : Number of neighbors for KNN
    :params distance_metric : Distance metric for KNN ('euclidean' or 'manhattan')
    :params n_folds         : Number of folds for cross-validation
    :params sklearn_check   : Using Sklearn KNN for comparison
    returns                 : mean_accuracy, std_accuracy
    """
    data = np.asarray(data)
    labels = np.asarray(labels)

    # Split data into n_folds partitions
    folds_data = np.array_split(data, n_folds, axis=0)
    folds_labels = np.array_split(labels, n_folds, axis=0)
    accuracies = []

    for fold in range(n_folds):
        # Select the test fold
        test_data = folds_data[fold]
        test_labels = folds_labels[fold]

        # Combine remaining folds for training
        train_data = np.concatenate([folds_data[i] for i in range(n_folds) if i != fold], axis=0)
        train_labels = np.concatenate([folds_labels[i] for i in range(n_folds) if i != fold], axis=0)

        if sklearn_check:
            # Use SciKit-Learn KNN
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            knn.fit(train_data, train_labels)  # Train the model
        else:
            # Use custom KNN
            knn = KNearestNeighbors(train_data, train_labels, k, distance_metric)

        # Compute accuracy
        correct = 0
        for x, true_label in zip(test_data, test_labels):
            if sklearn_check:
                pred_label = knn.predict(x.reshape(1, -1))[0]  # Ensure 2D input for sklearn
            else:
                pred_label = knn.predict(x)
            if pred_label == true_label:
                correct += 1
        accuracy = correct / len(test_labels)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)

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