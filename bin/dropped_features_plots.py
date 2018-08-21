import matplotlib
matplotlib.use('Agg')
from activ import load_data, data_normalization
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from activ.readfile import TrackTBIFile
from matplotlib.backends.backend_pdf import PdfPages

file = '/Users/ahyeon/Desktop/activ/results/mahalanobis_output/umap_cluster_uoinmf-mahalanobis.h5'
f = h5py.File(file, 'r')
score = f['score'][:]
norm_score = f['norm_score'][:]
clusters = f['clusters'][:]
cluster_sizes = f['cluster_sizes'][:]

scores = []
percent = [0,10,20,30,40,50]
for p in percent:
    fname = '/Users/ahyeon/Desktop/files/umap_cluster_features_{}.h5'.format(p)
    with h5py.File(fname, 'r') as f:
        score = f['score'][:]
        scores.append(score)

pdf = PdfPages('/Users/ahyeon/Desktop/activ/results/features_dropped/features_dropped_plot.pdf')
fig, ax = plt.subplots()
for s,p in zip(scores,percent):
    avg = s.mean(axis=0).mean(axis=0).mean(axis=1)
    totalavg = avg.mean(axis=0)
    print(totalavg, p)
    ax.plot(cluster_sizes, avg, '.', markersize='2', label='{}%'.format(p))
plt.xlabel('Cluster Sizes')
plt.ylabel('Accuracy')
ax.legend()
pdf.savefig()
pdf.close()
