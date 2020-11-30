import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize

from activ.nmf.viz import bases_heatmap
from activ import TrackTBIFile


def plot_bases(bases, colors, feat_names=None, return_groups=False):
    plt.figure(figsize=(56,28))
    ret = bases_heatmap(bases, aspect='auto',
                        col_labels=feat_names,
                        highlight_weight=0.50,
                        highlight=colors,
                        return_groups=return_groups)
    if return_groups:
        return ret[1], ret[3]
    return ret[1]


def plot_weights(weights, colors, row_order):
    plt.figure(figsize=(20, 10))
    ax = plt.gca()

    domfac = np.argmax(weights, axis=1)

    stacked = np.zeros((weights.shape[1], weights.shape[1]))
    bm_norm = normalize(weights)
    for b in row_order:
        subset = bm_norm[domfac == b]
        if subset.shape[0] == 0:
            continue
        stacked[b] = np.sum(subset, axis=0)/subset.shape[0]
    stacked = stacked[row_order][:, row_order]

    x = np.array(row_order)
    x = np.arange(weights.shape[1])
    for i in range(stacked.shape[1]):
        ax.bar(x, stacked[:, i], color=colors[i])
    ax.set_xticks(x)
    ret = ax.set_xticklabels(row_order)




parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='path to the TRACK-TBI file with CCA results')
parser.add_argument('-o', '--outdir', type=str, help='the directory to save figures to', default='.')
parser.add_argument('-f', '--format', type=str, choices=['png', 'pdf'], help='the directory to save figures to', default='pdf')
parser.add_argument('-t', '--feat_type', type=str, help='the feature class to plot', default=None)

args = parser.parse_args()

tbifile = TrackTBIFile(args.input)

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

# plot biomarker results
bm_colors = sns.color_palette('Set2', tbifile.nmf.bm_bases.shape[0])
bm_row_order = plot_bases(tbifile.nmf.bm_bases, bm_colors, feat_names=tbifile.biomarker_features)
plt.savefig(f'{args.outdir}/biomarker_bases.{args.format}')
plot_weights(tbifile.nmf.bm, bm_colors, bm_row_order)
plt.savefig(f'{args.outdir}/biomarker_weights.{args.format}')
bm_row_order, bm_var_grps = bm_row_order
bm_var_grps = pd.concat([pd.Series(data=grp) for grp in bm_var_grps], ignore_index=True, axis=1)
bm_var_grps.columns = [f'nmf_{i+1}' for i in bm_row_order]
bm_var_grps.to_csv(f'{args.outdir}/biomarker_groups.csv')

# plot outcome results
oc_colors = sns.color_palette('Set2', tbifile.nmf.oc_bases.shape[0])
oc_row_order = plot_bases(tbifile.nmf.oc_bases, oc_colors, feat_names=tbifile.outcome_features)
plt.savefig(f'{args.outdir}/outcome_bases.{args.format}')
plot_weights(tbifile.nmf.oc, oc_colors, oc_row_order)
plt.savefig(f'{args.outdir}/outcome_weights.{args.format}')
oc_row_order, oc_var_grps = oc_row_order
oc_var_grps = pd.concat([pd.Series(data=grp) for grp in oc_var_grps], ignore_index=True, axis=1)
oc_var_grps.columns = [f'nmf_{i+1}' for i in oc_row_order]
oc_var_grps.to_csv(f'{args.outdir}/outcome_groups.csv')
