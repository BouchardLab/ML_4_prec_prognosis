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

mds = MDS()
lr = LogisticRegression()

parser = get_parser()
args = parser.parse_args()

data = args.filepath
data.data_oc
data.feature_oc
data.data_bm
data.feature_bm
print("data_oc",data.data_oc.shape)
print("feature_oc", data.feature_oc.shape)
print("data_bm", data.data_bm.shape)
print("feature_bm", data.feature_bm.shape)

ranges = range(2,20)

cluster_output = cluster_range(data.data_oc, ranges)
# cluter_output.shape = (177,18)
print(cluster_output.shape)
score_clusters(data.feature_oc, cluster_output)
print(score_clusters)



# for n in range(1,11):
#     pca = PCA(n_components = n)
#     fa = FactorAnalysis(n_components = n)
#     method = best_decomp_method(data_matrix_subset_biomarker, pca, fa)
#     print(method)
