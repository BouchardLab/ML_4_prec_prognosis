import glob
import os
import numpy as np
import h5py

from activ.clustering.clustering import get_noc
from activ.clustering.summarize import read_clustering_results

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+', help='the files to summarize')
parser.add_argument('-o', '--output', type=str, default='simdata_sweep_results.h5', help='the output file')
parser.add_argument('--spread_asm', action='store_true', default=False, help='incorporate asymptote uncertainty when estimating noc')
parser.add_argument('--spread_foc', action='store_true', default=False, help='incorporate FOC uncertainty when estimating noc')
parser.add_argument('--use_median', action='store_true', default=False, help='use median when calculating FOC central tendancy')
parser.add_argument('--smooth', action='store_true', default=False, help='smooth data before fitting curve')

args = parser.parse_args()

true_noc = list()
est_noc_1sd = list()
est_noc_38ci = list()
est_noc_95ci = list()
est_noc_99ci = list()
est_ttest_5 = list()
est_ttest_1 = list()
iters = list()
for i, path in enumerate(args.files):
    print(i, path)
    noc, iter = list(map(int, path[:-7].split('_')[1:3]))

    true_noc.append(noc)
    iters.append(iter)
    _noc, _foc, _accuracy, _chance = read_clustering_results(path)
    kwargs= dict(noc=_noc,
                 foc=_foc,
                 plot=False,
                 fit_summary=True,
                 ttest_cutoff=False,
                 iqr=not args.smooth,
                 use_median=args.use_median,
                 spread_asm=args.spread_asm,
                 spread_foc=args.spread_foc)
    noc_idx = get_noc(n_sigma=1.0, **kwargs)
    est_noc_1sd.append(_noc[noc_idx])

    noc_idx = get_noc(ci=0.38, **kwargs)
    est_noc_38ci.append(_noc[noc_idx])

    noc_idx = get_noc(ci=0.95, **kwargs)
    est_noc_95ci.append(_noc[noc_idx])

    noc_idx = get_noc(ci=0.99, **kwargs)
    est_noc_99ci.append(_noc[noc_idx])

with h5py.File(args.output, 'w') as f:
    f.create_dataset('true_noc', data=true_noc)
    f.create_dataset('iters', data=iters)
    f.create_dataset('est_noc_1sd', data=est_noc_1sd)
    f.create_dataset('est_noc_38ci', data=est_noc_38ci)
    f.create_dataset('est_noc_95ci', data=est_noc_95ci)
    f.create_dataset('est_noc_99ci', data=est_noc_99ci)
    f.create_dataset('est_ttest_1', data=est_ttest_1)
    f.create_dataset('est_ttest_5', data=est_ttest_5)
