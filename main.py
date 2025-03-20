import numpy as np
from data.csv_parser import CSVParser
import src.validation as test


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
labels = np.array([1 if val >= 50 else 0 for val in csv_obj.get_column("track_popularity")])

# # =======================
# # Decision Tree Evaluation
# # =======================
# max_depth = 5
# num_thresh = 10
# mean_self_dt, std_self_dt = test.cross_validate_decision_tree(data, labels, max_depth, num_thresh, n_folds=5, sklearn_check=False)
# mean_sk_dt, std_sk_dt = test.cross_validate_decision_tree(data, labels, max_depth, n_folds=5, sklearn_check=True)

# # =======================
# # KNN Evaluation
# # =======================
# knn_k = 1
# mean_self_knn, std_self_knn = test.cross_validate_knn(data, labels, knn_k, n_folds=5, sklearn_check=False)
# mean_sk_knn, std_sk_knn = test.cross_validate_knn(data, labels, knn_k, n_folds=5, sklearn_check=True)

# # =======================
# # Results Output
# # =======================
# print("\n===== Model Evaluation Results =====")
# print("\nDecision Tree:")
# print(f"  - Custom Implementation -> Mean: {mean_self_dt:.4f}, Std: {std_self_dt:.4f}")
# print(f"  - Scikit-learn -> Mean: {mean_sk_dt:.4f}, Std: {std_sk_dt:.4f}")

# print("\nK-Nearest Neighbors (KNN):")
# print(f"  - Custom Implementation -> Mean: {mean_self_knn:.4f}, Std: {std_self_knn:.4f}")
# print(f"  - Scikit-learn -> Mean: {mean_sk_knn:.4f}, Std: {std_sk_knn:.4f}")


# =========================
# Neural Network Parameters
# =========================

# Format data in term of [# of samples, # of features]
indices = np.where(csv_obj.get_column("track_popularity") > 0)[0] 
cleaned_data = np.array([csv_obj.get_column(criteria) for criteria in test_criterias])[:, indices]
cleaned_labels = csv_obj.get_column("track_popularity")[indices]
cleaned_labels = np.array([1 if val >= 50 else 0 for val in cleaned_labels])


# cleaned_data = np.array([25, 20])
# cleaned_data = cleaned_data[:, np.newaxis]
# cleaned_labels = np.array([1])

layer       = 1
node        = 5
epochs      = 10
batch_size  = 128
mean_self, std_self, mean_sk, std_sk = 0, 0, 0, 0
mean_self, std_self = test.cross_validate_neural_network(cleaned_data.T, cleaned_labels, layer, 
                                                         epochs, batch_size, hidden_nodes=node)
# # mean_sk, std_sk     = cross_validate(data, label, params=max_depth, algo="tree", sklearn_check=True)
print(f"Self DT-> mean:{mean_self}, std: {std_self}\nSKLearn DT-> mean:{mean_sk}, std: {std_sk}")