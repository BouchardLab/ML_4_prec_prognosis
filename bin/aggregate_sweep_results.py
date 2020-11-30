import numpy as np
import h5py
import glob
import os

from activ.clustering.summarize import read_clustering_results, summarize_flattened, plot_line
from activ.clustering.clustering import flatten


import argparse

parser = argparse.ArgumentParser(description='aggregate clustering problem simulation results',
                                 epilog='paths should be of the form sim_<true_no_of_clusters>_<iteration_no>.h5')
parser.add_argument('output', help='output from UMAP-clustering pipeline')
parser.add_argument('files', nargs='+', help='output from UMAP-clustering pipeline')

args = parser.parse_args()

foc = list()
noc = list()
accuracy = list()
chance = list()
for path in sorted(args.files):
    _noc, iter = list(map(int, os.path.basename(path)[:-7].split('_')[1:3]))
    noc.append(_noc)
    _noc, _foc, _accuracy, _chance = read_clustering_results(path)
    foc.append(_foc)
    accuracy.append(_accuracy)
    chance.append(_chance)

def add_dset(g, name, data, desc):
    dset = g.create_dataset(name, data=data)
    dset.attrs['description'] = desc

with h5py.File(args.output, 'w') as f:
    add_dset(f, 'foc', foc, 'fold-over-chance accuracy')
    add_dset(f, 'accuracy', accuracy, 'predictive accuracy')
    add_dset(f, 'chance', chance, 'predictive accuracy from randomly shuffled labels')
    add_dset(f, 'noc', noc, 'the true number of clusters in each simulation result')

