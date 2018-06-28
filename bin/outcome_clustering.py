from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py
from activ.pipeline import filter_outliers, pca, best_decomp_method, cluster_range, score_clusters, score_clusters_cv
from activ.readfile import get_parser
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

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

ranges = np.asarray(range(2,50))
outcome_clustering_results = np.zeros((10, len(ranges)))

# without filtering outliers, pca
#print(np.transpose(data.data_oc).shape)
#cluster_output = cluster_range(np.transpose(data.data_oc), ranges, method='ward')
#print(cluster_output.shape)
#scores = score_clusters_cv(np.transpose(data.data_bm), cluster_output, cv=10)
#np.savez('outcome_cv_results.npz', name1 = scores)

# with filtering_outliers, pca
#new_ranges = np.asarray(range(2,12))
#data_oc_pca = pca(np.transpose(data.data_oc), 12)
#data_bm_pca = pca(np.transpose(data.data_bm), 12)
#data_oc_pca_filter, discard_indices = filter_outliers(data_oc_pca, 10)
#data_bm_pca_filter, discard_indices = filter_outliers(data_bm_pca, 10)
#print(data_oc_pca_filter.shape)
#cluster_output = cluster_range(data_oc_pca_filter, new_ranges, method='ward')
#print(cluster_output.shape)
#filtered_scores = score_clusters_cv(data_bm_pca_filter, cluster_output, cv=10)
#np.savez('outcome_cv_filtered_results.npz', name1 = filtered_scores)


plt.figure()
tree = hierarchy.linkage(data_oc_pca_filter, 'single')
hierarchy.dendrogram(tree)
plt.show()

