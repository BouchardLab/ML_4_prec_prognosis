from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py
from activ.pipeline import best_decomp_method, cluster_range, score_clusters
from activ.readfile import get_parser
import numpy as np


# biomarker_dec: NMF, ICA, DL (dictionary learning)
# outcome_dec: PCA, FA (factor analysis), MDS (multidim scaling), UMAP


parser = get_parser()
args = parser.parse_args()

data = args.filepath
data.data_oc
data.feature_oc
data.data_bm
data.feature_bm
#print("data_oc",data.data_oc.shape)
#print("feature_oc", data.feature_oc.shape)
#print("data_bm", data.data_bm.shape)
#print("feature_bm", data.feature_bm.shape)

ranges = np.asarray(range(2,200))
outcome_clustering_results = np.zeros((10, len(ranges)))

for i in range(10):
    cluster_output = cluster_range(np.transpose(data.data_oc), ranges)
    scores = score_clusters(np.transpose(data.data_bm), cluster_output)
    outcome_clustering_results[i,:] = scores
print(outcome_clustering_results)
print(outcome_clustering_results.shape)

np.savez('outcome_clustering_results.npz', name1 = outcome_clustering_results)

# for n in range(1,11):
#     pca = PCA(n_components = n)
#     fa = FactorAnalysis(n_components = n)
#     method = best_decomp_method(data_matrix_subset_biomarker, pca, fa)
#     print(method)
