import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from src.decision_tree import DecisionTree
from src.k_nn import KNearestNeighbors
from src.neural_n import NeuralNetwork

"""
=================================
Five-Fold Cross Validation (Split)
=================================
"""

def cross_validate_decision_tree(data, labels, max_depth, num_thresh=10, n_folds=5, sklearn_check=False):
    """
    Perform n-fold cross-validation for Decision Tree.
    :params data            : Training data (shape: [n_samples, n_features])
    :params labels          : Training labels (shape: [n_samples])
    :params max_depth       : Max depth for the decision tree
    :params n_folds         : Number of folds for cross-validation
    :params sklearn_check   : Use sklearn's DecisionTreeClassifier if True
    :returns                : mean_accuracy, std_accuracy
    """
    data            = np.asarray(data)
    labels          = np.asarray(labels)
    folds_data      = np.array_split(data, n_folds, axis=0)
    folds_labels    = np.array_split(labels, n_folds, axis=0)
    accuracies      = []

    for fold in range(n_folds):
        test_data, test_labels = folds_data[fold], folds_labels[fold]
        # Combine remaining folds for training
        train_data      = np.concatenate([folds_data[i] for i in range(n_folds) if i != fold], axis=0)
        train_labels    = np.concatenate([folds_labels[i] for i in range(n_folds) if i != fold], axis=0)

        if sklearn_check:
            clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
            clf = clf.fit(train_data, train_labels)
        else:
            clf = DecisionTree(train_data, train_labels, max_depth=max_depth, num_thresh=num_thresh)
            clf.build_tree()

        correct = 0
        for x, true_label in zip(test_data, test_labels):
            if sklearn_check == True:
                pred_label = clf.predict(x.reshape(1, -1))[0]
            else:
                pred_label = clf.predict(x)
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(test_labels)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


def cross_validate_knn(data, labels, k, n_folds=5, sklearn_check=False):
    """
    Perform n-fold cross-validation for K-Nearest Neighbors.
    :params data            : Training data (shape: [n_samples, n_features])
    :params labels          : Training labels (shape: [n_samples])
    :params k               : Number of neighbors for KNN
    :params n_folds         : Number of folds for cross-validation
    :params sklearn_check   : Use sklearn's KNeighborsClassifier if True
    :returns                : mean_accuracy, std_accuracy
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    folds_data = np.array_split(data, n_folds, axis=0)
    folds_labels = np.array_split(labels, n_folds, axis=0)
    accuracies = []

    for fold in range(n_folds):
        test_data, test_labels = folds_data[fold], folds_labels[fold]
        train_data = np.concatenate([folds_data[i] for i in range(n_folds) if i != fold], axis=0)
        train_labels = np.concatenate([folds_labels[i] for i in range(n_folds) if i != fold], axis=0)

        if sklearn_check:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf = clf.fit(train_data, train_labels)
        else:
            clf = KNearestNeighbors(train_data, train_labels, k)

        correct = 0
        for x, true_label in zip(test_data, test_labels):
            if sklearn_check == True:
                pred_label = clf.predict(x.reshape(1, -1))[0]
            else:
                pred_label = clf.predict(x)
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(test_labels)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


def cross_validate_neural_network(data, labels, hidden_layers, epochs, batch_size,
                                  hidden_nodes=5, output_size=1, n_folds=5):
    """
    Perform n-fold cross-validation for Neural Network.
    :params data            : Training data (shape: [n_samples, n_features])
    :params labels          : Training labels (shape: [n_samples])
    :params hidden_layers   : Number of hidden layers
    :params epochs          : Number of training epochs
    :parans batch_size      : Training batch size
    :params hidden_nodes    : Nodes in hidden layer
    :params output_size     : Output layer size
    :params n_folds         : Number of folds for cross-validation
    :returns                : mean_accuracy, std_accuracy
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    folds_data = np.array_split(data, n_folds, axis=0)
    folds_labels = np.array_split(labels, n_folds, axis=0)
    accuracies = []

    for fold in range(n_folds):
        test_data, test_labels = folds_data[fold], folds_labels[fold]
        train_data = np.concatenate([folds_data[i] for i in range(n_folds) if i != fold], axis=0)
        train_labels = np.concatenate([folds_labels[i] for i in range(n_folds) if i != fold], axis=0)

        clf = NeuralNetwork(np.shape(train_data)[1], hidden_layers, hidden_nodes, output_size)
        clf.train(train_data, train_labels, epochs, batch_size)

        correct = 0
        for x, true_label in zip(test_data, test_labels):
            pred_label = clf.predict(x)
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(test_labels)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)
