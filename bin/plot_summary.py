import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize

import matplotlib.patches as mpatches

from activ.nmf.viz import bases_heatmap
from activ import TrackTBIFile

def get_col(m, feats, colname):
    col = None
    idx = np.where(feats == varname)[0]
    if len(idx) > 0:
        idx = idx[0]
        col = m[:, idx]
    return col

def add_name(func):
    def ret(ax, col):
        func(ax, col)
        ax.set_title(col.name)
    return ret


@add_name
def make_barplot(ax, col):
    vals, counts = np.unique(col, return_counts=True)
    ax.bar([str(v) for v in vals], counts, color='gray')

@add_name
def make_barplot_ints(ax, col):
    vals, counts = np.unique(col, return_counts=True)
    ax.bar([str(int(v)) for v in vals], counts, color='gray')

@add_name
def make_age(ax, col):
    counts, edges = np.histogram(col, bins=np.arange(1,9)*10)
    names = [f'{b}-{e}' for b, e in zip(edges[:-1], edges[1:])]
    ax.bar(names, counts, color='gray')

@add_name
def make_gcs(ax, col):
    vals, counts = np.unique(col, return_counts=True)
    vals = vals.astype(int)
    mask = vals >= 13
    ax.bar(vals[mask], counts[mask], color='blue', label='Mild')
    mask = np.logical_and(np.logical_not(mask), vals >= 9)
    ax.bar(vals[mask], counts[mask], color='green', label='Moderate')
    mask = np.logical_and(np.logical_not(mask), vals < 9)
    ax.bar(vals[mask], counts[mask], color='red', label='Severe')
    ax.set_xticks(np.arange(1,8)*2+1)
    ax.set_xticklabels(np.arange(1,8)*2+1)
    ax.legend()

@add_name
def make_yes_no(ax, col):
    vals, counts = np.unique(col, return_counts=True)
    vals = ['No', 'Yes']
    ax.bar(vals, counts, color='gray')


def make_race(ax, df):
    ar = np.zeros(len(df), dtype='U5')
    ar[df['RACE_3CAT_Black'] == 1] = 'Black'
    ar[df['RACE_3CAT_White'] == 1] = 'White'
    ar[df['RACE_3CAT_Other'] == 1] = 'Other'
    make_barplot(ax, pd.Series(data=ar, name='Race'))

@add_name
def make_hist(ax, col):
    ax.hist(col, bins=40, color='gray')

parser = argparse.ArgumentParser()
parser.add_argument('tbifile', help='the TRACK-TBI file to use', type=TrackTBIFile)
parser.add_argument('-o', '--outdir', type=str, help='the directory to save figures to', default='.')
parser.add_argument('-f', '--format', type=str, choices=['png', 'pdf'], help='the directory to save figures to', default='pdf')
args = parser.parse_args()

tbifile = args.tbifile



df = pd.DataFrame(data=np.concatenate([tbifile.biomarkers, tbifile.outcomes], axis=1),
                  columns=np.concatenate([tbifile.biomarker_features, tbifile.outcome_features]),
                  index=tbifile.patient_ids)

figure, axes = plt.subplots(2, 5, figsize=(25, 10))
make_age(axes[0, 0], df['Age'])
make_race(axes[0, 1], df)
make_gcs(axes[0, 2], df['admGCS'])
make_yes_no(axes[0, 3], df['ICPMonitorYesNo'])
make_yes_no(axes[0, 4], df['PMH_Psych_ANY'])

make_barplot_ints(axes[1, 0], df['GOSE_OverallScore3M'])
make_barplot_ints(axes[1, 1], df['RPQNausea_6mo'])
make_barplot_ints(axes[1, 2], df['NeuroOverallRating3mo'])
make_hist(axes[1, 3], df['CVLTTrial1To5StandardScore_6mo'])
make_yes_no(axes[1, 4], df['PTSD_6mo'])

plt.savefig(f'{args.outdir}/tracktbi_summary.{args.format}')
