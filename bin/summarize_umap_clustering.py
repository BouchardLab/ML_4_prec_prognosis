import glob
import os
import numpy as np
import h5py

from activ.clustering import read_data, get_noc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, default='simdata_sweep_results.h5', help='the output file')
parser.add_argument('--spread_asm', action='store_true', default=False, help='incorporate asymptote uncertainty when estimating noc')

args = parser.parse_args()

true_noc = list()
est_noc_1sd = list()
est_noc_38ci = list()
est_noc_95ci = list()
est_noc_99ci = list()
est_ttest_5 = list()
est_ttest_1 = list()
iters = list()
for i, path in enumerate(glob.glob('*.out.h5')):
    print(i, path)
    noc, iter = list(map(int, path[:-7].split('_')[1:3]))

    true_noc.append(noc)
    iters.append(iter)
    _noc, _foc, _accuracy, _chance = read_data(path)

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=False, n_sigma=1.0, spread_asm=args.spread_asm)
    est_noc_1sd.append(_noc[noc_idx])

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=False, ci=0.38, spread_asm=args.spread_asm)
    est_noc_38ci.append(_noc[noc_idx])

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=False, ci=0.95, spread_asm=args.spread_asm)
    est_noc_95ci.append(_noc[noc_idx])

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=False, ci=0.99, spread_asm=args.spread_asm)
    est_noc_99ci.append(_noc[noc_idx])

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=True, pvalue_cutoff=0.05, spread_asm=args.spread_asm)
    est_ttest_5.append(_noc[noc_idx])

    noc_idx = get_noc(_noc, _foc, plot=False, fit_summary=True, ttest_cutoff=True, pvalue_cutoff=0.01, spread_asm=args.spread_asm)
    est_ttest_1.append(_noc[noc_idx])

with h5py.File(args.output, 'w') as f:
    f.create_dataset('true_noc', data=true_noc)
    f.create_dataset('iters', data=iters)
    f.create_dataset('est_noc_1sd', data=est_noc_1sd)
    f.create_dataset('est_noc_38ci', data=est_noc_38ci)
    f.create_dataset('est_noc_95ci', data=est_noc_95ci)
    f.create_dataset('est_noc_99ci', data=est_noc_99ci)
    f.create_dataset('est_ttest_1', data=est_ttest_1)
    f.create_dataset('est_ttest_5', data=est_ttest_5)
