import seaborn as sns
import itertools as it
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.patches as mpatches

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


def get_labels(column):
    uniq = np.unique(column)
    colors = sns.hls_palette(len(uniq))
    lut = dict(zip(uniq, colors))
    labels = pd.Series([lut[v] for v in column])
    patches = [mpatches.Patch(color=c, label=l) for l, c in lut.items()]
    return labels, patches


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
            cumsum_thresh=0.99,
            cbar_kw={}, cbarlabel="", xlab=None, ylab=None,
            title=None, **kwargs):
    """
    Create a heatmap from outcome factors.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
    Optional arguments:
        row_labels : A list or array of length N with the outcome
                     factors for the rows
        col_labels : A list or array of length M with the outcome
                     labels for the columns
        sort       : whether or not to sort rows and columns. If a array-like is passed in,
                     it will be used to order rows.
        xlab       : the label for the x-axis
        ylab       : the label for the y-axis
        title      : the title for the heatmap
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    kwargs.setdefault('cmap', 'binary')
    if row_labels is None or col_labels == True:
        row_labels = np.arange(data.shape[0]) + 1
    if col_labels is None or col_labels == True:
        col_labels = np.arange(data.shape[1]) + 1


    plot_data = np.array(data)


    row_order = None
    col_order = None
    if sort is not False:
        if sort is not True:
            factor_order = np.asarray(sort)
        else:
            factor_order = np.argsort(np.max(plot_data, axis=1))[::-1]
        plot_data = plot_data[factor_order]
        to_sort = list()
        for i, c in enumerate(plot_data.T):
            mi = np.argmax(c)
            to_sort.append((mi, -1*c[mi], i))
        new_order = np.array([t[2] for t in sorted(to_sort)])
        plot_data = plot_data[:,new_order]
        row_order = factor_order
        col_order = new_order

    for i in range(plot_data.shape[0]):
        order = np.argsort(plot_data[i])[::-1]
        cumsum = np.cumsum(plot_data[i,order])
        cumsum = cumsum / cumsum[-1]
        plot_data[i][order[cumsum > cumsum_thresh]] = 0.0

    # Plot the heatmap
    im = ax.imshow(plot_data, **kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    if col_labels is not False:
        ax.set_xticks(np.arange(plot_data.shape[1]))
        col_labels = np.asarray(col_labels)[col_order]
        ax.set_xticklabels(col_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if row_labels is not False:
        ax.set_yticks(np.arange(plot_data.shape[0]))
        row_labels = np.asarray(row_labels)[row_order]
        ax.set_yticklabels(row_labels)
    else:
        ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)

    ax.set_xlabel(None)
    if xlab is not None:
        ax.set_xlabel(xlab,fontsize=36)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=36)
    if title is not None:
        ax.set_title(title, fontsize=48)

    # Rotate the tick labels and set their alignment.
    return im, row_order, col_order


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


def nmfplot(weights, bases, hm_kwargs=None, cm_kwargs=None, indiv_lev_frac=None, total_lev_frac=None):
    sig_bases = None
    if indiv_lev_frac is not None:
        sig_bases = get_sig_indiv_lev(bases, frac=indiv_lev_frac)
    if total_lev_frac is not None:
        sig_bases = get_sig_total_lev(bases, frac=total_lev_frac)
    if sig_bases is not None:
        bases = bases[:,sig_bases]

    kwargs = dict()
    if cm_kwargs is not None:
        kwargs.update(cm_kwargs)
    cg, row_linkage, col_linkage = weights_clustermap(weights, **kwargs)
    bases_order = sch.leaves_list(col_linkage)

    kwargs = dict(sort=bases_order)
    if hm_kwargs is not None:
        kwargs.update(hm_kwargs)
    im, row_order, col_order = bases_heatmap(bases, **kwargs)

    return cg, im, row_linkage, col_linkage, row_order, col_order

