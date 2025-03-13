import numpy as np

from src.decision_tree import DecisionTree
from data.csv_parser import CSVParser
from src.k_nn import cross_validate_knn


path = "data/data.csv"
csv_obj = CSVParser(path)
test_criterias = ["danceability", "energy", "key", "loudness", "mode", "speechiness", 
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
                  "duration_ms"]

data = {}
for criteria in test_criterias:
    data[criteria] = np.array(csv_obj.get_column(criteria))

c_k = [1 if val >= 50 else 0 for val in csv_obj.get_column("track_popularity")]
# tree = DecisionTree(data=data, attributes=test_criterias, label=c_k)
# tree.build_tree()

train_data = np.array([data[criteria] for criteria in test_criterias])
c_k = np.array(c_k)

sub_data = train_data.T
sub_c_k = c_k
mean, std = cross_validate_knn(sub_data, sub_c_k, k=1, distance_metric='euclidean', n_folds=5, sklearn_check=False)
print(mean, std)

