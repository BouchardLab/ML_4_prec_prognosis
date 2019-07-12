import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.colors as mpc


def get_labels(column):
    uniq = np.unique(column)
    colors = sns.hls_palette(len(uniq))
    lut = dict(zip(uniq, colors))
    labels = pd.Series([lut[v] for v in column])
    patches = [mpatches.Patch(color=c, label=l) for l, c in lut.items()]
    return labels, patches


def multi_stemplot(values, ax=None, labels=None, pallette=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    colors = [mpc.to_hex(c) for c in sns.color_palette(pallette, values.shape[0])]
    shift = np.max(np.abs(values))*2.1
    for i in range(values.shape[0]):
        ax.stem(values[i]+i*shift, markerfmt=' ', basefmt=colors[i], linefmt=colors[i], bottom=i*shift)
    pos = np.arange(values.shape[0]) * shift
    ax.get_xaxis().set_ticks([])
    yaxis = ax.get_yaxis()
    yaxis.set_ticks(pos)
    ticklabels = np.arange(values.shape[0])
    if labels is not None:
        ticklabels = labels
    yaxis.set_ticklabels(ticklabels)
