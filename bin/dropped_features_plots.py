import matplotlib
matplotlib.use('Agg')
from activ import load_data, data_normalization
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from activ.readfile import TrackTBIFile


file = '/Users/ahyeon/Desktop/activ/results/mahalanobis_output/umap_cluster_uoinmf-mahalanobis.h5'
f = h5py.File(file, 'r')
score = f['score'][:]
norm_score = f['norm_score'][:]
clusters = f['clusters'][:]
cluster_sizes = f['cluster_sizes'][:]

scores = []
percent = [160,20,30,40,50]
for p in percent:
    fname = '/Users/ahyeon/Desktop/activ/results/features_dropped/umap_cluster_features_{}.h5'.format(p)
    f = h5py.File(fname, 'r')
    score = f['score'][:]
    scores.append(score)
print(scores.shape)
