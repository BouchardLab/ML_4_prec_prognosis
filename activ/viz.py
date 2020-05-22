import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.lines as mlines
import matplotlib.colors as mpc

from sklearn.linear_model import LinearRegression

from .analytics import cv_r2_score, linefit, _check_X_y


def get_labels(column, palette='hls', marker=None, solid_points=False,
               marker_kwargs=dict()):
    uniq = np.unique(column)
    colors = sns.color_palette(palette, len(uniq)).as_hex()[::-1]
    lut = dict(zip(uniq, colors))
    labels = pd.Series([lut[v] for v in column])
    _kwargs = dict()
    _kwargs.update(marker_kwargs)
    if marker is not None:
        _kwargs['marker'] = marker
    if solid_points:
        patches = [mlines.Line2D([0], [0], label=l, mec=c, mfc=c, **_kwargs) for l,c in lut.items()]
    else:
        patches = [mlines.Line2D([0], [0], label=l, mec=c, **_kwargs) for l,c in lut.items()]
    return labels, patches


def multi_stemplot(values, ax=None, labels=None, palette='Set1'):
    if len(values.shape) == 1:
        values = values.reshape(1, -1)
    if ax is None:
        ax = plt.gca()
    colors = [mpc.to_hex(c) for c in sns.color_palette(palette, values.shape[0])]
    shift = np.max(np.abs(values))*2.1
    for i in range(values.shape[0]):
        ax.stem(values[i]+i*shift,
                use_line_collection=True,
                markerfmt=' ',
                basefmt=colors[i],
                linefmt=colors[i],
                bottom=i*shift)
    pos = np.arange(values.shape[0]) * shift
    ax.get_xaxis().set_ticks([])
    yaxis = ax.get_yaxis()
    yaxis.set_ticks(pos)
    ticklabels = np.arange(values.shape[0])
    if labels is not None:
        ticklabels = labels
    yaxis.set_ticklabels(ticklabels)


def _sort_by_count(ar):
    """
    A helper function for fitplot. Return unique values sorted by count.
    """
    unique, count = np.unique(ar, return_counts=True)
    unique = unique[np.argsort(count)][::-1]
    return unique


def fitplot(x, y, regressor=LinearRegression(), labels=None, fitline=True,
            solid_points=False, marker='o', opacity=None,
            title=None, xlabel=None, ylabel=None, legend_title=None, ax=None,
            r2_display='raw', fontsize=12):
    """
    Scatter plot with best-fit line drawn and R-squared displayed.
    """
    if ax is None:
        ax = plt.gca()
    _x, _y = _check_X_y(x, y)

    colors, patches = None, None
    if labels is not None:
        colors, patches = get_labels(labels, palette='hls')
    else:
        colors = ['black'] * _x.shape[0]

    if opacity is not None:
        opacity = np.array(opacity)
        unique_opacity = np.sort(np.unique(opacity))
        alphas = np.linspace(0.2, 1.0, unique_opacity.shape[0])
        tmp = np.zeros(len(opacity))
        for op, a in zip(unique_opacity, alphas):
            tmp[opacity == op] = a
        opacity = tmp
    else:
        opacity = np.array([1] * x.shape[0])

    if solid_points:
        ax.scatter(p_x, p_y, c=p_c, s=5, marker=m)
    else:
        # sort by count, so we plot the densest one first
        for c in _sort_by_count(colors):
            mask = colors == c
            p_x, p_y, p_c, p_o = _x[mask], _y[mask], colors[mask], opacity[mask]
            for o in _sort_by_count(p_o):
                mask2 = p_o == o
                p_x2, p_y2, p_c2 = p_x[mask2], p_y[mask2], p_c[mask2]
                ax.scatter(p_x2, p_y2, edgecolors=p_c2, facecolors='none', s=150, linewidths=6, marker=marker, alpha=o)

    if patches is not None:
        ax.legend(handles=patches, title=legend_title, loc=2, prop={'size': int(fontsize*2/3)}, markerfirst=True)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict={'fontsize':fontsize})
        ax.xaxis.set_label_coords(0.5, -0.12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict={'fontsize':fontsize})
        ax.yaxis.set_label_coords(-0.15, 0.5)

    if fitline:
        r2_raw, P_raw, r2_cv, P_cv = cv_r2_score(regressor, _x, _y)
        xfit, yfit = linefit(regressor, _x, _y)
        ax.plot(xfit, yfit, color='black', linestyle='-', linewidth=4)
        ax.set_xlim(min(xfit), max(xfit))
        ax.set_ylim(min(yfit), max(yfit))
        tmpl = "$R^2$ ({r2_display}) = %0.4f\nP < e%d"
        r2, P = None, None
        if r2_display == 'raw':
            r2, P = r2_raw, P_raw
        elif r2_display == 'cv':
            r2, P = r2_cv, P_cv
        if r2 is not None:
            tmpl = tmpl.format(r2_display=r2_display)
            power = int(np.ceil(np.log10(P_cv)))
            ax.text(0.5, 0.1, tmpl % (r2, power), size=fontsize, transform=ax.transAxes)
    return ax


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.binary):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
