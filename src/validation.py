import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from src.decision_tree import DecisionTree
from src.k_nn import KNearestNeighbors
from src.neural_n import NeuralNetwork

"""
============================
Five Fold Cross Validation
============================
"""
def cross_validate(data, labels, params, n_folds=5, algo = "tree", sklearn_check=False):
    """
    Perform n-fold cross-validation for the object and report mean and std of accuracy
    for the folds.
    
    :params data        : Training data (shape: [n_samples, n_features])
    :params labels      : Training labels (shape: [n_samples])
    :params algo        : Machine learning algorithm
    :params n_folds     : Number of folds for cross-validation
    :params params      : Parameters for the algorithm
    :returns            : mean_accuracy, std_accuracy
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

        if sklearn_check is True:
            if algo == "tree":
                clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=params)
            if algo == "knn":
                clf = KNeighborsClassifier(n_neighbors=params)
            clf = clf.fit(data, labels)
            
        else:
            if algo == "tree":
                clf = DecisionTree(train_data, train_labels, max_depth=params)
                clf.build_tree()
            if algo == "knn":
                clf = KNearestNeighbors(train_data, train_labels, params)
            if algo == "nn":
                clf = NeuralNetwork(np.shape(train_data)[1], hidden_layers=params, hidden_nodes=5, output_size=1)
                clf.train(train_data, train_labels, epochs=10, batch_size=1)

        # Compute accuracy
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