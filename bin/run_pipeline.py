from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py
from activ.pipeline import best_decomp_method
import argparse
from activ.readfile import ReadFile_obj

# biomarker_dec: NMF, ICA, DL (dictionary learning)
# outcome_dec: PCA, FA (factor analysis), MDS (multidim scaling), UMAP

mds = MDS()
lr = LogisticRegression()


parser = argparse.ArgumentParser()
parser.add_argument('filepath', type = ReadFile_obj)
args = parser.parse_args()

data = args.filepath
print(data.data_bm.shape)



# for n in range(1,11):
#     pca = PCA(n_components = n)
#     fa = FactorAnalysis(n_components = n)
#     method = best_decomp_method(data_matrix_subset_biomarker, pca, fa)
#     print(method)