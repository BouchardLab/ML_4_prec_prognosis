import numpy as np
from scipy.cluster import hierarchy
from activ.pipeline import filter_outliers, run_umap
import h5py
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, cross_val_score

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
        for ii,k in enumerate(umap_dims): # umap dimension
            embedding = run_umap(test_data, k, n_neighbors=10, min_dist=0.000, random_state=1)
            Z = hierarchy.linkage(embedding, 'ward', optimal_ordering=True)
            trees.append(Z)
            for jj,c in enumerate(clusters): # cluster
                labels = hierarchy.cut_tree(Z, n_clusters=c).reshape(237)
                score[x,ii,jj] = cross_val_score(RFC(100, n_jobs=n_jobs), test_data, labels, cv=10)
                norm_score[x,ii,jj] = cross_val_score(RFC(100, n_jobs=n_jobs), test_data, np.random.permutation(labels), cv=10)
        embdict[x] = embeddings
        treedict[x] = trees
        print(x)
#     fname = '/Users/ahyeon/Desktop/TBIanalysis/umap_{}_results.h5'.format(dataname)
    fname = '/home/ahyeon/data/umap_{}_results.h5'.format(dataname)
    with h5py.File(fname, 'w') as f:
        f.create_dataset('score', data=score)
        f.create_dataset('norm_score', data=norm_score)
#     dd.io.save('/Users/ahyeon/Desktop/TBIanalysis/{}_embeddings.h5'.format(dataname), embdict)
#     dd.io.save('/Users/ahyeon/Desktop/TBIanalysis/{}_trees.h5'.format(dataname), treedict)
    dd.io.save('/home/ahyeon/data/{}_embeddings.h5'.format(dataname), embdict)
    dd.io.save('/home/ahyeon/data/{}_trees.h5'.format(dataname), treedict)
    return score, norm_score, embdict, treedict

oc_W1 = '/home/ahyeon/data/data_matrix_subset_outcome.h5'

oc_w1 = read_h5(oc_W1)
oc_w1, indices = filter_outliers(oc_w1,10)

score, norm_score, emb, tree = umap_pipeline(50, oc_w1, 'oc_w1', 8)
