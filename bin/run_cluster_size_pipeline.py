import numpy as np
from scipy.cluster import hierarchy
from activ.pipeline import filter_outliers, run_umap
import h5py
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, cross_val_score
import deepdish as dd
import warnings

def read_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    return data.T

def umap_pipeline(iteration, test_data, dataname, n_jobs=1):
    umap_dims = range(2,11)#11
    clusters = range(2,15) #15
    score = np.zeros((iteration, len(umap_dims), len(clusters), 10))
    norm_score = np.zeros((iteration, len(umap_dims), len(clusters), 10))
    embdict = {}
    treedict = {}

    for x in range(iteration):
        embeddings = []
        trees = []
        print ("--------- iteration %s ---------" % x)
        for ii,k in enumerate(umap_dims): # umap dimension
            embedding = run_umap(test_data, k, n_neighbors=10, min_dist=0.000)
            Z = hierarchy.linkage(embedding, 'ward', optimal_ordering=True)
            trees.append(Z)
            for jj,c in enumerate(clusters): # cluster
                print('umap dimensions: %s, num clusters: %s' % (k, c))
                labels = hierarchy.cut_tree(Z, n_clusters=c).reshape(237)
                warnings.filterwarnings("ignore")
                score[x,ii,jj] = cross_val_score(RFC(100, n_jobs=n_jobs), test_data, labels, cv=5)
                norm_score[x,ii,jj] = cross_val_score(RFC(100, n_jobs=n_jobs), test_data, np.random.permutation(labels), cv=10)
        embdict[x] = embeddings
        treedict[x] = trees
    fname = '/home/ahyeon/data/cv_results_{}_{}.h5'.format(dataname,iteration)
    with h5py.File(fname, 'w') as f:
        f.create_dataset('score', data=score)
        f.create_dataset('norm_score', data=norm_score)
    dd.io.save('/home/ahyeon/data/embeddings_{}_{}.h5'.format(dataname,iteration), embdict)
    dd.io.save('/home/ahyeon/data/trees_{}_{}.h5'.format(dataname,iteration), treedict)
    return score, norm_score, embdict, treedict

oc_W1 = '/home/ahyeon/data/data_matrix_subset_outcome.h5'
bm_W1 = '/home/ahyeon/data/data_matrix_subset_biomarker.h5'
oc_w1 = read_h5(oc_W1)
bm_w1 = read_h5(bm_W1)
oc_w1, indices = filter_outliers(oc_w1,10)
bm_w1, indices = filter_outliers(bm_w1,10)

umap_pipeline(1, bm_w1, 'bm_w1', 8)
umap_pipeline(50, bm_w1, 'bm_w1', 8)
