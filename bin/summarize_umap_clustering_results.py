import matplotlib
matplotlib.use('Agg')
import h5py
import numpy as np
from activ.analytics import heatmap
from activ.clustering import UmapClusteringResults
from os.path import dirname, join
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Create summary pdf of umap clustering results.')
parser.add_argument('path', type=str, help='path to h5 file')
parser.add_argument('filename', type=str, help='name of file')
args = parser.parse_args()

#path = "../results/mahalanobis_output/umap_cluster_uoinmf-mahalanobis.h5"
#pdf = PdfPages(join(dirname(path), 'umap_clustering_sweep_mahalanobis_summary.pdf'))

pdf = PdfPages(join(dirname(args.path), args.filename))
results = UmapClusteringResults(args.path)
plt.figure(figsize=(15, 5))
ret = results.make_heatmap()
pdf.savefig()

plt.figure(figsize=(12,8))
plt.title("Mean accuracy by cluster size across\nall iterations and dimension size")
plt.xlabel("Cluster sizes")
plt.ylabel("Fold over chance")
ret = plt.boxplot(results.adjust(), labels=results.cluster_sizes)
pdf.savefig()

pdf.close()
