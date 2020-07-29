from pyuoi.decomposition.NMF import UoI_NMF, UoI_NMF_Base
from sklearn.cluster import KMeans
from activ.readfile import TrackTBIFile
from activ.utils import get_logger
from activ.data_normalization import data_normalization
from datetime import  datetime
from sklearn.utils import check_random_state
import numpy as np

import h5py
import sys
import os
import argparse
from time import time

def run(X, args, k=None):
    if not args.no_norm:
        X = data_normalization(X, 'positive')
    random_state = check_random_state(args.seed)
    if k is not None:
        nmf = UoI_NMF_Base(n_boots=args.n_boots, ranks=[k], cluster=KMeans(k))
    else:
        nmf = UoI_NMF(args.n_boots, ranks=range(6,9), nmf_init='random', db_min_samples=2)
    nmf.fit(X)
    W = nmf.transform(X)
    H = nmf.components_
    return W, H, str(nmf)

def get_nmf_feat_type(H):
    idx = np.argmax(H, axis=0)
    feat = ['nmf_%d' % (i+1) for i in idx]
    return {'nmf': feat}


parser = argparse.ArgumentParser(usage="%(prog)s [options] <hdf5_path>:<dataset_path> <output_hdf5_path>")
parser.add_argument('input', type=str, help='the path to the TRACK-TBI input file')
parser.add_argument('-b', '--n_boots', type=int, help='the number of bootstraps to run', default=20)
parser.add_argument('-s', '--seed', type=int, help='the seed to use for random number generation', default=-1)
parser.add_argument('-n', '--no_norm', action='store_true', help='do not normalize data to [0,1]', default=False)
parser.add_argument('--no_outcomes', action='store_true', help='do not run UoI-NMF on outcome data', default=False)
parser.add_argument('-O', '--outcomes_k', type=int, help='number of latent outcome dimensions', default=None)
parser.add_argument('-B', '--biomarkers_k', type=int, help='number of latent biomarkers dimensions', default=None)


args = parser.parse_args()

args.seed = args.seed if args.seed >= 0 else int(round(time() * 1000) % 2**32)
logger = get_logger()
logger.info('using seed %s' % args.seed)

logger.info('loading data')
tbifile = TrackTBIFile(args.input)

logger.info('running UoI-NMF on biomarkers')
bm_w, bm_h, bm_nmf = run(tbifile.biomarkers, args, args.biomarkers_k)
n, q = tbifile.outcomes.shape
if not args.no_outcomes:
    logger.info('running UoI-NMF on outcomes')
    oc_w, oc_h, oc_nmf = run(tbifile.outcomes, args, args.outcomes_k)
else:
    logger.info('skipping outcomes')
    oc_w, oc_h, oc_nmf = np.zeros((n, 5)), np.zeros((5, q)), 'not run'

metadata = {'biomarker_params': str(bm_nmf), 'outcome_params': str(oc_nmf)}

logger.info('computing NMF types')
bm_ft = get_nmf_feat_type(bm_h)
oc_ft = get_nmf_feat_type(oc_h)

logger.info('writing results')
TrackTBIFile.write_nmf(args.input, bm_w, oc_w, bm_h, oc_h, metadata=metadata, overwrite=True)
TrackTBIFile.write_feat_types(args.input, bm_ft, oc_ft, overwrite=True)
