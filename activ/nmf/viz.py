import seaborn as sns
import itertools as it
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.stats as st
from sklearn.preprocessing import normalize

import matplotlib.axes as mpl_axes
import matplotlib.colors as mpc
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import bases_factor_order
from ..viz import get_labels

def get_comb(order):
    """Return a linkage matrix that will order labels in the order in which they are given"""
    Z = list()
    nclus = len(order)
    tmp = order[::-1]
    Z.append([tmp[1], tmp[0], 1, 2])
    for i in range(nclus-2):
        Z.append([tmp[i+2], nclus+i, 2+i, 2+1])
    Z = np.array(Z, dtype=float)
    return Z


def get_sig_total_lev(nmf_bases, frac=0.5):
    nmf_leverage = nmf_bases.sum(axis=0)
    order = np.argsort(nmf_leverage)[::-1]
    cumsum = np.cumsum(nmf_leverage[order])
    cumsum = cumsum/cumsum[-1]
    idx = np.where(cumsum > frac)[0][0]
    min_lev = nmf_leverage[order[idx]]
    sig = np.where(nmf_leverage > min_lev)[0]
    return sig


def get_sig_indiv_lev(nmf_bases, frac=0.250):
    indices = get_sig_variables(nmf_bases, frac=frac)
    return np.unique(list(it.chain.from_iterable(indices)))


def get_sig_variables(nmf_bases, frac=0.250):
    indices = list()
    for b in range(nmf_bases.shape[0]):
        basis = nmf_bases[b]
        order = np.argsort(basis)[::-1]
        cumsum = np.cumsum(basis[order])
        cumsum = cumsum/cumsum[-1]
        idx = np.where(cumsum > frac)[0][0]
        indices.append(order[:idx])
    return indices


def bases_heatmap(data, col_labels=False, row_labels=False, sort=True, ax=None,
            highlight=False, highlight_weight=0.7, cumsum_thresh=0.99, show_cbar=True,
            cbar_kw={}, cbarlabel="", xlabel=None, ylabel=None, fontsize=None,
            title=None, return_groups=False, factor_order=None, **kwargs):
    """
    Create a heatmap from outcome factors.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
    Optional arguments:
        row_labels         : A list or array of length N with the outcome
                             factors for the rows
        col_labels         : A list or array of length M with the outcome
                             labels for the columns
        sort               : whether or not to sort rows and columns. If a array-like is passed in,
                             it will be used to order rows.
        highlight          : True to highlight bases with different colorwise. Custom colors
                             can be used by passing in a custom pallete. False to not color
                             (default = False)
        highlight_weight   : the weight of the higlighting (default = 0.7)
        xlab               : the label for the x-axis
        ylab               : the label for the y-axis
        title              : the title for the heatmap
        ax                 : A matplotlib.axes.Axes instance to which the heatmap
                             is plotted. If not provided, use current axes or
                             create a new one.
        cbar_kw            : A dictionary with arguments to
                             :meth:`matplotlib.Figure.colorbar`.
        cbarlabel          : The label for the colorbar
        kwargs             : additional arguments to pass in to imshow
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    kwargs.setdefault('cmap', 'binary')
    if row_labels is None or row_labels is True:
        row_labels = np.arange(data.shape[0]) + 1
    if col_labels is None or col_labels is True:
        col_labels = np.arange(data.shape[1]) + 1


    plot_data = np.array(data)

    if factor_order is None:
        factor_order = bases_factor_order(plot_data)

    plot_data = plot_data[factor_order]
    to_sort = list()
    for i, c in enumerate(plot_data.T):
        mi = np.argmax(c)
        to_sort.append((mi, -1*c[mi], i))
    to_sort = sorted(to_sort)
    feat_order = np.array([t[2] for t in to_sort])
    plot_data = plot_data[:,feat_order]

    row_order = factor_order
    col_order = feat_order

    for i in range(plot_data.shape[0]):
        order = np.argsort(plot_data[i])[::-1]
        cumsum = np.cumsum(plot_data[i,order])
        cumsum = cumsum / cumsum[-1]
        plot_data[i][order[cumsum > cumsum_thresh]] = 0.0

    kwargs.setdefault('interpolation', 'none')
    # Plot the heatmap
    im = ax.imshow(plot_data, **kwargs)


    # Create colorbar
    if show_cbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, cax=cax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)

    if col_labels is not False:
        ax.set_xticks(np.arange(plot_data.shape[1]))
        col_labels = np.asarray(col_labels)[col_order]
        ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if row_labels is not False:
        row_labels = np.asarray(row_labels)[row_order]
        row_labels = [""] + row_labels.tolist()
        ax.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        ax.set_yticklabels(row_order, fontsize=fontsize)

    ax.set_xlabel(None)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    label_groups = None
    if highlight is not None and highlight is not False:

        if isinstance(highlight, bool):
            highlight = sns.color_palette('Set2', plot_data.shape[0])
        highlight = [highlight[i] for i in factor_order]
        maxes = np.argmax(plot_data, axis=1)
        bnd = list()
        for b_i in range(len(maxes)-1):
            mx = maxes[b_i+1]
            if mx < maxes[b_i]:
                srt = np.argsort(plot_data[b_i+1])
                i = 1
                new_mx = srt[i]
                while new_mx <= maxes[b_i]:
                    i += 1
                    new_mx = srt[i]
                bnd.append(new_mx)
            else:
                bnd.append(mx)
        label_groups = np.array(bnd+[plot_data.shape[1]])
        bnd = [b - 0.5 for b in bnd]
        bounds = list(zip([-0.5]+bnd, bnd+[plot_data.shape[1]]))
        y = -0.5
        height = plot_data.shape[1]
        for b_i, bound in enumerate(bounds):
            x = bound[0]
            width = bound[1] - x
            color = highlight[b_i]
            ax.add_patch(mpatches.Rectangle((x, y), width, height, fill=True,
                         facecolor=color, lw=0, alpha=highlight_weight))

    ret = [im, row_order, col_order]
    if return_groups and label_groups is not None:
        if col_labels is False or col_labels is None:
            col_labels = [str(i) for i in range(data.shape[1])]
        s = 0
        tmp_lblgrp = list()
        for e in label_groups:
            tmp_lblgrp.append(col_labels[s:e])
            s = e
        ret.append(tmp_lblgrp)
    # Rotate the tick labels and set their alignment.
    return tuple(ret)


def get_percent_top(features, bases, perc=0.80, return_perc=False):
    """
    For each basis, return top *perc* features according to contribution
    to the basis

    Args:
        return_perc (bool)        : return percents along with features

    Returns:
        a list of lists or a tuple of lists of lists if *return_perc* is *True*

    """
    ret = list()
    percents = list()
    for b in range(bases.shape[0]):
        basis = bases[b]
        order = np.argsort(basis)[::-1]
        idx = np.where(np.cumsum(basis[order]/basis.sum()) >= perc)[0][0] + 1
        ret.append(features[order][:idx])
        if return_perc:
            percents.append(basis[order][:idx]/basis.sum())
    if return_perc:
        return tuple(ret), tuple(percents)
    return tuple(ret)


def plot_bases(bases, colors, feat_names=None, return_groups=True, ax=None,
               bases_labels=None, factor_order=None, fontsize=None, xlabel=None):
    """
    A wrapper for *bases_heatmap*
    """
    ret = bases_heatmap(bases, aspect='auto',
                        col_labels=feat_names,
                        highlight_weight=0.50,
                        highlight=colors,
                        show_cbar=True,
                        # cbarlabel='Feature contribution',
                        return_groups=return_groups,
                        row_labels=bases_labels,
                        xlabel=xlabel,
                        factor_order=factor_order,
                        fontsize=fontsize,
                        ax=ax)
    if return_groups:
        return ret[1], ret[3]
    return ret[1]


def plot_weights(weights, colors=None, factor_order=None, ax=None, labels=None, fontsize=None, labelsize=None):
    """
    Plot weights as a barplot.

    For each weight, take the average of all patients with that weight as their maximum weight, and order
    weights according to the max weight
    """
    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = sns.color_palette('Set2', weights.shape[1])

    labelsize = labelsize or fontsize

    domfac = np.argmax(weights, axis=1)
    stacked = np.zeros((weights.shape[1], weights.shape[1]))
    bm_norm = normalize(weights, norm='l1')
    dist = np.zeros(weights.shape[1], dtype=int)
    for b in range(weights.shape[1]):
        mask = domfac == b
        dist[b] = mask.sum()
        subset = bm_norm[mask]
        if subset.shape[0] == 0:
            continue
        stacked[b] = np.mean(subset, axis=0)
    #############

    if factor_order is None:
        factor_order = np.argsort(np.max(stacked, axis=0))[::-1]

    stacked = stacked[factor_order][:, factor_order]
    x = np.arange(weights.shape[1])

    if not isinstance(ax, mpl_axes.Axes):
        if len(ax) > 1:
            # ax[0].axis('off')
            ax_bar = ax[0]
            ax_bar.tick_params('x', labelbottom=False, bottom=False)
            ax_bar.bar(x, dist[factor_order]/dist.sum(), color='Grey')
            yticks = np.array([0.4, 0.8])
            ax_bar.set_yticks(yticks)
            ax_bar.set_yticklabels(yticks, fontsize=labelsize)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.set_ylabel('Fraction of\npatients', fontsize=fontsize)
            ax = ax[1]
        else:
            ax = ax[0]



    y_offset = 0.0
    colors = [colors[i] for i in factor_order]
    for i in range(stacked.shape[1]):
        ax.bar(x, stacked[:, i], bottom=y_offset, color=colors[i])
        y_offset += stacked[:, i]

    no_dom = np.where(np.logical_not(stacked.sum(axis=1).astype(bool)))[0]
    if len(no_dom) > 0:
        ax.bar(x[no_dom], np.ones(len(no_dom)), fc='none', ec='black', hatch='//')


    ax.set_xticks(x)
    if labels is None:
        labels = factor_order
    else:
        labels = [labels[i] for i in factor_order]
    ax.set_xticklabels(labels, rotation=45, rotation_mode='anchor', horizontalalignment='right', fontsize=labelsize)
    yticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticks * 100).astype(int), fontsize=labelsize)
    ax.set_ylim(0, 1.02)

    ax.set_ylabel('Average Percent of Total Weight', fontsize=fontsize)
    return factor_order


def nmfplot(weights, bases, palette=None, features=None, axes=None, bases_labels=None,
            bases_order=True, legend=False, fontsize=None, heatmap_xlabel=None, labelsize=None):
    """
    Generate a side-by-side figure of a bases heatmap and a weights barplot
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    if palette is None:
        palette = sns.color_palette('Set2', bases.shape[0])
    labelsize = labelsize or fontsize
    if bases_order:
        factor_order, var_grps = plot_bases(bases, palette, feat_names=features or False,
                                            ax=axes[0], bases_labels=bases_labels, fontsize=labelsize)
        axes[0].set_xlabel(heatmap_xlabel, fontsize=fontsize)
        plot_weights(weights, palette, factor_order=factor_order, ax=axes[1:], labels=bases_labels, fontsize=fontsize, labelsize=labelsize)
    else:
        factor_order = plot_weights(weights, palette, ax=axes[1:], labels=bases_labels, fontsize=fontsize, labelsize=labelsize)
        factor_order, var_grps = plot_bases(bases, palette, factor_order=factor_order, fontsize=labelsize,
                                            feat_names=features or False, ax=axes[0], bases_labels=bases_labels)
        axes[0].set_xlabel(heatmap_xlabel, fontsize=fontsize)
    if bases_labels is not None and legend:
        axes[1].legend([mpatches.Patch(color=c) for c in palette], bases_labels, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    return factor_order


def weights_pie_scatter(weights, emb, s=100, ax=None, palette=None):
    """
    Plot NMF weights piecharts

    Args:
        weights (array)         : the NMF weights
        emb (array)             : 2D embedding of patients
    """
    if palette is None:
        palette = sns.color_palette('Set2', weights.shape[1])

    ax = ax or plt.gca()
    weights = np.cumsum(weights, axis=1)
    weights = 2 * np.pi * weights/np.repeat(weights.max(axis=1), weights.shape[1]).reshape(weights.shape)

    for p_i in range(weights.shape[0]):
        start = 0
        for w_i in range(weights.shape[1]):
            end = weights[p_i, w_i]
            lin = np.linspace(start, end, 10)
            start = end
            x = [0] + np.cos(lin).tolist()
            y = [0] + np.sin(lin).tolist()
            xy = np.column_stack([x, y])
            ax.scatter([emb[p_i, 0]], [emb[p_i, 1]], marker=xy, s=s, color=palette[w_i])


def plot_umap_nmf_piechart(weights, umap_emb, s=100, ax=None, fontsize=None, palette=None):
    """
    Plot 2-D UMAP embedding as pie-charts with distribution of weights
    """
    weights_pie_scatter(weights, umap_emb, s=s, ax=ax, palette=palette)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('UMAP dimesion 1', fontsize=fontsize)
    ax.set_ylabel('UMAP dimesion 2', fontsize=fontsize)


def plot_umap_nmf_max(emb, weights, bases_labels, right=False, min_dist=0.0, legend=True, ax=None, palette=None):
    """
    Plot 2-D UMAP embedding, colored according to max NMF weight
    """
    ax = ax or plt.gca()

    sf = np.argmax(weights, axis=1)
    if palette is None:
        palette = sns.color_palette('Set2', weights.shape[1])
    colors = np.zeros((weights.shape[0], 3), dtype=float)
    size = np.zeros(weights.shape[0], dtype=float)
    total = np.sum(weights, axis=1)
    for f in np.unique(sf):
        mask = sf == f
        colors[mask] = palette[f]
        size[mask] =  weights[mask, f]/total[mask]
    size /= size.max()
    size *= 100

    uniq, counts = np.unique(sf, return_counts=True)
    uniq = uniq[np.argsort(counts)[::-1]]
    handles = list()
    labels = list()
    for f in uniq:
        mask = sf == f
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[mask], s=size[mask], label=bases_labels[f], edgecolors='w')
        handles.append(mlines.Line2D([0], [0], marker='o', color='w', label=bases_labels[f],
                                     markerfacecolor=colors[mask][0], markersize=10))
        labels.append(bases_labels[f])

    if legend:
        if right:
            plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        else:
            plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0, 1.0))


def get_point_colors(cmap, vec):
    mappable = cm.ScalarMappable(cmap=cmap,
                                 norm=mpc.Normalize(vmin=np.min(vec), vmax=np.max(vec)))
    colors = np.array([mappable.to_rgba(_) for _ in vec])
    return colors, mappable


def plot_umap_nmf_weight(emb, weights, axes, bases_labels, cmaps='Reds'):
    """
    Plot 2-D UMAP embedding, one for each weight, coloring points according
    to the value of the weight
    """
    if isinstance(cmaps, (str, mpc.Colormap)):
        cmaps = [cmaps] * weights.shape[1]
    for ax, vec, label, cmap in zip(axes, weights.T, bases_labels, cmaps):
        colors, mappable = get_point_colors(plt.get_cmap(cmap), vec)
        ax.scatter(emb[:, 0], emb[:, 1], c=colors)
        cax = make_axes_locatable(ax).append_axes("right", size="7%", pad="2%")
        plt.colorbar(mappable, cax=cax)
        ax.tick_params('both', labelsize='large')
        cax.tick_params('both', labelsize='large')
        ax.set_title(label, fontsize='x-large')
    #    ax.axis('scaled')
    for ax in axes[weights.shape[1]:]:
        ax.axis('off')


def plot_umap_nmf_weight_kde(emb, weights, colors, bases_labels=None, cbar=True, alpha=1.0, ax=None, scatter=False, scatter_kw=None):
    """
    Plot smoothed 2D histogram of weights across UMAP embeddings.
    """

    if ax is None:
        ax = plt.gca()

    x = emb[:, 0]
    y = emb[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    adj = 0.2
    x_adj = (xmax - xmin)*adj
    y_adj = (ymax - ymin)*adj

    xmin, xmax = xmin - x_adj, xmax + y_adj
    ymin, ymax = ymin - y_adj, ymax + y_adj

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    add_cbar = True
    if isinstance(ax, plt.Axes):
        ax = [ax] * weights.shape[1]
        add_cbar = False

    add_cbar = cbar and add_cbar

    fig = ax[0].figure

    alphas = weights.sum(axis=0)
    alphas /= alphas.sum()

    for axes, vec, color, sub_alpha in zip(ax, weights.T, colors, alphas):
        if alpha is not None:
            sub_alpha = alpha
        kernel = st.gaussian_kde(values, bw_method='scott', weights=vec)
        f = np.reshape(kernel(positions).T, xx.shape)

        cmap = mpc.LinearSegmentedColormap.from_list('mycmap', [(1.0,1.0,1.0), color])
        levels = np.linspace(0, np.max(f)*1.1, 110)
        cfset = axes.contourf(xx, yy, f, levels, cmap=cmap, alpha=sub_alpha)
        if scatter:
            tmp = vec - vec.min()
            tmp = tmp / tmp.max()
            if scatter_kw is None:
                scatter_kw = dict()
            scatter_kw.pop('fc', None)
            scatter_kw.pop('alpha', None)
            axes.scatter(x, y, fc=color, alpha=tmp*sub_alpha, **scatter_kw)
        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymin, ymax)
        if add_cbar:
            fig.colorbar(cfset, ax=axes)

    if bases_labels is not None:
        for axes, label in zip(ax, bases_labels):
            axes.set_title(label, fontsize='x-large')

    for axes in ax[weights.shape[1]:]:
        axes.axis('off')


def cumulative_plot(bases, labels, ax=None, colors=None, title=None, mark=None):
    """
    Make plot of cumulative contribution of variables to bases
    """
    ax = ax or plt.gca()
    if colors is None:
        colors = sns.color_palette("Set2", bases.shape[0])

    cumsum = (np.cumsum(bases, axis=1).T / np.sum(bases, axis=1)).T
    for i in range(cumsum.shape[0]):
        ax.plot(cumsum[i], color=colors[i], label=labels[i])
        if mark is not None:
            if mark > 1:
                mark = 1/100
            feat = np.where(cumsum[i] > 0.8)[0][0]
            ax.axvline(feat, color='r', ls=':', zorder=0)
    fs = 'xx-large'
    ax.legend(fontsize='x-large', loc='lower right')
    ax.tick_params(labelsize=fs)
    ax.set_xticks([])
    ax.set_xlabel(title, fontsize=fs)
    ax.set_ylabel('Cumulative leverage', fontsize=fs)


def weights_clustermap(data, row_linkage=None, col_linkage=None, cmap='binary',
                       row_labels=None, legend_title="", **cm_kwargs):
    if row_linkage is None:
        row_linkage = sch.linkage(data, method='ward')
    if col_linkage is None:
        col_linkage = sch.linkage(data.T, method='ward')
    kwargs = dict(
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap=cmap,
    )
    label_legend = dict()
    kwargs['row_colors'] = list()
    if row_labels is not None:
        if isinstance(row_labels, dict):
            for k, v in row_labels.items():
                lbl, leg = get_labels(v)
                kwargs['row_colors'].append(lbl)
                label_legend[k] = leg
        else:
            lbl, leg = get_labels(row_labels)
            kwargs['row_colors'].append(lbl)
            label_legend[legend_title] = leg

    if cm_kwargs is not None:
        kwargs.update(cm_kwargs)
    cg = sns.clustermap(data, **kwargs)

    st = 1.2          # TODO: figure out how to not have to hardcode this!
    l2 = None
    for ll_name, ll in label_legend.items():
        if l2 is not None:
            cg.ax_col_dendrogram.add_artist(l2)
        l2 = cg.ax_col_dendrogram.legend(loc='best', bbox_to_anchor=(st,1.0), handles=ll, frameon=True)
        l2.set_title(ll_name, prop={'size':8})
        st = st + 0.15

    return cg, row_linkage, col_linkage
