import numpy as np

from src.decision_tree import DecisionTree
from src.k_nn import KNearestNeighbors
from src.neural_n import NeuralNetwork as NNET
from data.csv_parser import CSVParser

from src.validation import cross_validate


"""
================
Processing CSV
================
"""
path = "data/data.csv"
csv_obj = CSVParser(path)
test_criterias = ["danceability", "energy", "key", "loudness", "mode", "speechiness", 
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
                  "duration_ms"]

# Format data in term of [# of samples, # of features]
data  = np.array([csv_obj.get_column(criteria) for criteria in test_criterias]).T
label = np.array([1 if val >= 50 else 0 for val in csv_obj.get_column("track_popularity")])

"""
========================
Decision Tree Parameters
========================
"""
# max_depth = 3
# mean_self, std_self = 0, 0
# mean_self, std_self = cross_validate(data, label, params=max_depth, algo="tree", sklearn_check=False)
# mean_sk, std_sk     = cross_validate(data, label, params=max_depth, algo="tree", sklearn_check=True)
# print(f"Self DT-> mean:{mean_self}, std: {std_self}\nSKLearn DT-> mean:{mean_sk}, std: {std_sk}")

"""
=========================
Neural Network Parameters
=========================
"""
hidden_layer = 0
mean_self, std_self, mean_sk, std_sk = 0, 0, 0, 0
mean_self, std_self = cross_validate(data[0:100, :], label[0:100], params=hidden_layer, algo="nn", sklearn_check=False)
# mean_sk, std_sk     = cross_validate(data, label, params=max_depth, algo="tree", sklearn_check=True)
print(f"Self DT-> mean:{mean_self}, std: {std_self}\nSKLearn DT-> mean:{mean_sk}, std: {std_sk}")

"""
==============
KNN Parameters
==============
"""
# knn_k       = 3
# n_folds     = 5 
# mean_self, std_self   = cross_validate(data, label, params=knn_k, n_folds=n_folds, algo="knn", sklearn_check=False)
# mean_sk, std_sk       = cross_validate(data, label, params=knn_k, n_folds=n_folds, algo="knn", sklearn_check=True)
# print(f"Self KNN-> mean:{mean_self}, std: {std_self}\nSKLearn KNN-> mean:{mean_sk}, std: {std_sk}")

