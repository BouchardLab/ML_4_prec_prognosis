from PyUoI.UoINMF import UoINMF
from activ import load_data, data_normalization
import matplotlib
matplotlib.use('Agg')
from activ import load_data, data_normalization
import matplotlib.pyplot as plt
import os
from activ.readfile import TrackTBIFile
import h5py
import numpy as np
from activ.analytics import heatmap, nmf_bases_heatmap
from activ.clustering import UmapClusteringResults
from os.path import dirname, join
from scipy.optimize import curve_fit
from scipy.stats import norm, sem

percent = [0,10,20,30,40,50,60,70,80,90]
percent2=[0,20,40,60,80,90]
for ii,p in enumerate(percent2):
    path = '/Users/ahyeon/data/activ/oc_features_dropped/oc_features_dropped_result_{}_5.h5'.format(p)
    f = h5py.File(path, 'r')
    score = f['score'][:]
    norm_score = f['norm_score'][:]
    cluster_sizes = f['cluster_sizes'][:]
    cluster_sizes.shape
    norm_score = norm_score.mean(axis=2).mean(axis=0).repeat(50).reshape(49,50).T
    scorem = score.mean(axis=2)
    std = np.std(scorem, axis=0)
    score = score.mean(axis=2)
    foc = score/norm_score
    points = foc.mean(axis=0)
    std = np.std(foc, axis=0)
    plt.rcParams["figure.figsize"] = [16,9]
    cmap = plt.get_cmap("tab10")
    plt.plot(cluster_sizes, points, label='{}%'.format(p), color=cmap(ii))
    plt.fill_between(cluster_sizes, points+std, points-std, alpha=0.1)
    plt.rcParams.update({'errorbar.capsize': 5})
    plt.title("Mean accuracy by cluster size across all iterations\nOutcome Features Dropped\nEuclidean")
    plt.xlabel("Cluster sizes")
    plt.ylabel("Fold over chance")
    plt.legend()
plt.savefig('/Users/ahyeon/Desktop/dim_collapsed.pdf')
plt.show()
