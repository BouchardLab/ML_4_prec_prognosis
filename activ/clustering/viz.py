import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mpc

import matplotlib.tri as tri
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from sklearn.linear_model import LinearRegression


from .summarize import flatten, flatten_summarize, read_clustering_results, summarize_flattened, plot_line
from .clustering import get_noc, get_noc_max1d_smooth


def make_simdata_curves(ax=None, path=None, keep_true_nocs=[10, 15, 20, 25, 30, 40], fontsize='x-large'):
    """
    Plot fold-over-chance (FOC) for simulation data.  Plot lines for simulations that used
    a true number-of-clusters (NOC) in *keep_true_nocs*

    Y-axis is normalized to be between (0, L), where L is the max FOC for that simulated model

    Args:
        path (str)              : the path to the aggregated simulation results file
        ax (plt.Axes)           : the Axes object to use for plot
        keep_true_nocs (bool)   : the number-of-clusters of the simulated models to plot
    """
    ax = ax or plt.gca()
    keep_true_nocs = np.array(keep_true_nocs)
    with h5py.File(path, 'r') as f:
        foc = f['foc'][:]
        nocs = f['noc'][:]
        mask = np.isin(nocs, keep_true_nocs)
        foc = foc[mask]
        nocs = nocs[mask]

        if 'tested_noc' in f:
            tested_nocs = f['tested_noc'][:]
        else:
            tested_nocs = np.arange(2,51)
    true_nocs = np.sort(np.unique(nocs))[::-1]
    X = np.zeros((foc.shape[0], foc.shape[1]))
    y = np.zeros(foc.shape[0], dtype=int)
    X_i = 0

    for j, true_noc in enumerate(true_nocs):
        idx = nocs == true_noc
        n = np.sum(idx)
        _noc = nocs[idx]
        _foc = foc[idx]
        for i in range(_foc.shape[0]):
            noc_flat, foc_flat = flatten(tested_nocs, _foc[i], filter_inf=False, smooth=True)

            _x, _lower, _med, _upper = summarize_flattened(noc_flat, foc_flat, iqr=False)
            _med /= np.max(_med)
            X[X_i] = _med
            y[X_i] = true_noc
            X_i += 1
    pal = sns.color_palette('Paired', len(keep_true_nocs)*2)[1::2]
    colors = dict(zip(keep_true_nocs, pal))

    #for X_i in range(X.shape[0]):
    tmpX = X[:, :-3]

    for i in np.unique(y)[::-1]:
        mask = y == i
        ax.plot(tmpX[mask].T, c=colors[i], label=i)

    handles = list()
    labels = list()
    for noc, col in colors.items():
        handles.append(mlines.Line2D([0], [0], color=col))
        labels.append(str(noc))

    ax.tick_params('both', labelsize=fontsize)
    ax.set_xlabel('# of clusters used for prediction', fontsize=fontsize)
    ax.set_ylabel("Fraction of Max Prediction Accuracy", fontsize=fontsize)
    ax.legend(handles, labels, title='True # of clusters', fontsize=fontsize, title_fontsize=fontsize, loc='lower right')


def get_link_color_func(Z, hex_labels):
    """
    Function for getting the colors of links drawn in a dendrogram
    """
    dflt_col = "#000000"
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
      c1, c2 = (link_cols[x] if x > len(Z) else hex_labels[x] for x in i12)
      link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col
    return lambda x: link_cols[x]


def shade_around(data, ax=None, stretch=0.05, alpha=0.2, color='blue'):
    """
    Draw a triangulated patch around a set of data points

    Args:
        data (np.array)         : the points to draw around
        ax (plt.Axes)           : the Axes to draw on
        stretch (float)         : the amount to stetch the triangle vertices
                                  from the center of the points
        alpha (float)           : the transparency of the patch
        color (str or tuple)    : the color of the patch

    """
    ax = ax or plt.gca()

    x = data[:, 0]
    y = data[:, 1]
    center = (x.mean(), y.mean())
    x = (x - center[0])*(1+stretch) + center[0]
    y = (y - center[1])*(1+stretch) + center[1]

    z = np.ones_like(x)


    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi = np.linspace(xmin, xmax, 2000)
    yi = np.linspace(ymin, ymax, 2000)

    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    #cfset = ax.contourf(xi, yi, zi, colors=color, alpha=alpha)
    cfset = ax.tricontourf(x, y, z, colors=color, alpha=alpha)


def cluster_results_plot(emb, colors, ax=None):
    """
    Plot 2-D UMAP embedding with colors

    Args:
        emb (np.array)          : 2-D UMAP embedding
        colors (np.array)       : array of colors for each point
    """
    ax = ax or plt.gca()
    for color in np.unique(colors):
        mask = colors == color
        shade_around(emb[mask], ax=ax, color=color, stretch=0.12)
    ax.scatter(emb[:,0], emb[:,1], c='k', edgecolors='w')


def dendrogram_scatterplot(axes, Z, emb, labels, ticks=True):
    """
    Plot 2-D UMAP embedding with a dendrogram juxtaposed

    Args:
        axes (plt.Axes)           : the Axes to draw the plots on
        Z (np.array)              : the dendrogram linkage matrix
        labels (np.array)         : the labels from cutting the linkage matrix
    """

    cluster_results_plot(emb, labels, ax=axes[0])

    ret = dendrogram(Z, color_threshold=None, show_leaf_counts=True,
                     no_labels=True, orientation='right', above_threshold_color='black',
                     link_color_func=get_link_color_func(Z, labels), ax=axes[1])
    if not ticks:
        for ax in axes:
            for xlabel_i in ax.axes.get_xticklabels():
                xlabel_i.set_visible(False)
            for tick in ax.axes.get_xticklines():
                tick.set_visible(False)
            for ylabel_i in ax.axes.get_yticklabels():
                ylabel_i.set_visible(False)
            for tick in ax.axes.get_yticklines():
                tick.set_visible(False)
    c = len(np.unique(labels))
    axes[1].axvline(x=(Z[-c+1,2] + Z[-c,2])/2, color='grey', lw=4, linestyle=':')

def make_dendrogram_scatterplots(emb, nmf_path=None, cmap='Dark2', axes=None):
    """
    Plot two 2-D UMAP embeddings with dendrogram on the side. The upper plot
    is cut at 3 clusters and the lower plot is at 4 clusters.
    """
    dist = pdist(emb)
    Z = linkage(dist, method='ward')
    labels = cut_tree(Z, n_clusters=[3, 4])
    if isinstance(cmap, str):
        pal = sns.color_palette(cmap, 10).as_hex()
        pal = [pal[i] for i in (3,1,0,2)]
    else:
        pal = cmap
    #pal = [pal[i] for i in ()]
    colors = np.zeros((len(emb), 2), dtype='U7')
    colors[labels[:,0] == 0, 0] = pal[0]
    colors[labels[:,0] == 1, 0] = pal[1]
    colors[labels[:,0] == 2, 0] = pal[2]

    colors[labels[:,1] == 0, 1] = pal[0]
    colors[labels[:,1] == 1, 1] = pal[1]
    colors[labels[:,1] == 2, 1] = pal[3]
    colors[labels[:,1] == 3, 1] = pal[2]

    if axes is None:
        fig, axes = plt.subplots(2,2)

    dendrogram_scatterplot(axes[0], Z, emb, colors[:, 0])
    dendrogram_scatterplot(axes[1], Z, emb, colors[:, 1])


def sim_sweep_plot(noc_est, true_nocs, violin=True, ax=None, data_below=False,
                   title=None, xlabel='True # of Clusters', ylabel='Estimated # of Clusters',
                   real_data_est=None, flip=False, fontsize='x-large'):
    """
    Plot true number-of-clusters vs predicted number-of-clusters from simulation sweep

    Args:
        noc_est (np.array)          : 2d array with shape (n_noc_tested, n_repetitions)
        true_nocs (np.array)        : 1d array with shape (n_noc_tested)
        data_below (bool)           : data underneath line fit
        real_data_est (int)         : the estimate from real data. if provided, draw horizontal
                                      line at the given number
        flip (bool)                 : plot predicted on x-axis. default is to plot true on x-axis
    """
    ax = ax or plt.gca()


    ######################
    # BEGIN: plot line fit
    ######################
    X = true_nocs.repeat(10)
    Y = noc_est.ravel()
    if flip:
        X, Y = Y, X
        xlabel, ylabel = ylabel, xlabel
    lr = LinearRegression().fit(X.reshape(-1, 1), Y)

    ##########################
    # Compute uncerainty band
    sd_Y = Y.std()
    y_est = lr.predict(X.reshape(-1, 1))

    r2 = lr.score(X.reshape(-1, 1), Y) #, y_est)
    y_err = X.std() * np.sqrt(1/len(X) +
                          (X - X.mean())**2 / np.sum((X - X.mean())**2))

    ax.plot(X, y_est, color='red', zorder=2)
    ax.fill_between(X.squeeze(), y_est + y_err, y_est - y_err, color='red', alpha=0.2, zorder=1)
    ####################
    # END: plot line fit
    ####################

    ####################
    # Plot line of unity
    ax.plot(true_nocs, true_nocs, ls='--', color='gray', lw=0.75)


    ################################
    # Add text with line fit results
    mn = np.min(true_nocs)
    mx = np.max(true_nocs)
    xpos = (mx - mn) * 0.3 + mn
    mn = np.min(noc_est)
    mx = np.max(noc_est)
    ypos = (mx - mn) * 0.85 + mn

    model_text = r"$y \approx %0.4fx + %0.4f$"
    if lr.intercept_ < 0.0:
        model_text = r"$y \approx %0.4fx %0.4f$"
    model_text = model_text % (lr.coef_[0], lr.intercept_)
    text = ("\n"
            r"$R^2 \approx %0.4f$")  % r2
    ax.text(xpos, ypos, model_text + text, fontsize=fontsize)

    #########################################
    # BEGIN: Plot violin plot or scatter plot
    #########################################
    data_zorder = 0 if data_below else 3
    if violin:
        ret = ax.violinplot(noc_est.T, positions=true_nocs, showextrema=False, showmeans=True, vert=not flip)
        segs = ret['cmeans'].get_segments()
        for seg in segs:
            seg[0,0] -= 0.5
            seg[1,0] += 0.5
        ret['cmeans'].set_segments(segs)
        ret['cmeans'].set_linewidth(0.5)
        ret['cmeans'].set_edgecolor('k')
        for pc in ret['bodies']:
            pc.set_facecolor('gray')
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(1)
            pc.set_zorder(data_zorder)
    else:
        ret = ax.scatter(X, Y, color='black', s=5, zorder=data_zorder)
    #######################################
    # END: Plot violin plot or scatter plot
    #######################################

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params('both', labelsize=fontsize)

    #############################
    # Add estimate from real data
    if real_data_est is not None:
        ax.axhline(real_data_est, ls='--', color='green')

    return lr

def _check_criterion(criterion):
    choices = ('1sd', '38ci', '95ci', '99ci')
    if criterion not in choices:
        msg = ("unrecognized criterion: '%s' "
               "-- must one of the following: %s" % (criterion, ", ".join(choices)))
        raise ValueError(msg)
    return 'est_noc_%s' % criterion

def _replace_zeros(ar):
    """Helper function for plot_simdata_curvefit_results"""
    ar[ar==0] = np.median(ar[ar!=0])
    return ar


def plot_simdata_curvefit_results(path, criterion='1sd', ax=None, trim=True, flip=False, fontsize='x-large'):
    """
    Read simulation data results and plot true number-of-clusters vs
    predicted number-of-clusters from simulation sweep
    """
    ax = ax or plt.gca()

    dset_name = _check_criterion(criterion)

    with h5py.File(path, 'r') as f:
        est_noc = f[dset_name][:]
        true_noc = f['true_noc'][:]

    uniq = np.sort(np.unique(true_noc))

    est_noc_a = np.zeros((uniq.shape[0], 10))

    for i, noc in enumerate(uniq):
        mask = true_noc == noc
        est_noc_a[i] = _replace_zeros(est_noc[mask])

    if trim:
        uniq = uniq[10:-10]
        est_noc_a = est_noc_a[10:-10]

    return sim_sweep_plot(est_noc_a, uniq, ax=ax, flip=flip, fontsize=fontsize)


def plot_max1d_simdata_results(path, ax=None, flip=False, fontsize='x-large'):
    """
    Plot max first derivative sweep results
    """
    ax = ax or plt.gca()
    with h5py.File(path, 'r') as f:
        est_noc_max1d = f['est_noc_max1d'][:]
        true_noc = f['true_noc'][:]
    uniq = np.sort(np.unique(true_noc))
    sim_sweep_plot(est_noc_max1d, true_noc, ax=ax, flip=flip, fontsize=fontsize)


def make_clustered_plot(emb, n_clusters, ax=None, cmap='tab10'):
    """
    Plot 2-D UMAP embedding. Cluster embedding and shade around the
    resulting clusters
    """
    ax = ax or plt.gca()
    dist = pdist(emb)
    Z = linkage(dist, method='ward')
    labels = cut_tree(Z, n_clusters=[n_clusters])
    labels = labels[:,0]
    if isinstance(cmap, str):
        pal = sns.color_palette(cmap, n_clusters).as_hex()
    else:
        pal = cmap

    colors = np.zeros(len(emb), dtype='U7')
    for cl_l in np.arange(n_clusters):
        mask = labels == cl_l
        colors[mask] = pal[cl_l]

    cluster_results_plot(emb, colors, ax=ax)

def get_real_noc(tested_noc, foc, smooth=True, use_median=False, spread_asm=True, spread_foc=True):
    ret = dict()
    kwargs= dict(noc=tested_noc,
                 foc=foc,
                 plot_fit=False,
                 fit_summary=True,
                 ttest_cutoff=False,
                 iqr=not smooth,
                 use_median=use_median,
                 spread_asm=spread_asm,
                 spread_foc=spread_foc)
    noc_idx = get_noc(n_sigma=1.0, **kwargs)
    ret['est_noc_1sd'] = tested_noc[noc_idx]

    noc_idx = get_noc(ci=0.38, **kwargs)
    ret['est_noc_38ci'] = tested_noc[noc_idx]

    noc_idx = get_noc(ci=0.95, **kwargs)
    ret['est_noc_95ci'] = tested_noc[noc_idx]

    noc_idx = get_noc(ci=0.99, **kwargs)
    ret['est_noc_99ci'] = tested_noc[noc_idx]

    return ret


def correct_estimate(est, path, correction_dset='1sd', trim=True):
    """
    Correct estimate according to linear model built from simulation data

    Args:
        est (int)       : the estimate to correct
        path (str)      : the path to the simulation data to use for the linear model
    """
    correction_dset = _check_criterion(correction_dset)
    with h5py.File(path, 'r') as f:
        est_noc = f[correction_dset][:]
        true_noc = f['true_noc'][:]

    uniq = np.sort(np.unique(true_noc))

    est_noc_a = np.zeros((uniq.shape[0], 10))
    for i, noc in enumerate(uniq):
        mask = true_noc == noc
        est_noc_a[i] = _replace_zeros(est_noc[mask])

    est_noc = est_noc_a
    if trim:
        uniq = uniq[10:-10]
        est_noc = est_noc[10:-10]

    y = uniq.repeat(10)
    X = est_noc.ravel().reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    y_est = lr.predict(X)
    y_err_stdev = np.sqrt(np.sum((y_est - y)**2)/(len(y) - 2))
    return int(np.round(lr.predict(np.array([[est]]))[0])), int(np.round(1.96*y_err_stdev))


def plot_real_foc_results(path, ax=None, max1d_cutoff=False, ci=None, n_sigma=1, simdata_results_path=None, fontsize='x-large'):
    """
    Plot fold-over-chance results for real data.

    Draws the following:
        - a scatter of test NOCs vs FOC, with error bars at each point
        - a dashed line for the exponential curve fit
        - a vertical line at the estimate, with uncertainty around it

    """
    ax = ax or plt.gca()
    real_noc, real_foc, real_accuracy, real_chance = read_clustering_results(path)

    real_data_est = get_real_noc(np.arange(2, 51), real_foc,
                                 smooth=True, use_median=False,
                                 spread_asm=True, spread_foc=True)


    x_noc, lower, med, upper = flatten_summarize(np.arange(2, 51),
                                                 real_foc, smooth=True,
                                                 iqr=False)

    plot_line(x_noc, med, lower=lower, upper=upper, ax=ax)

    if max1d_cutoff:
        noc_idx = get_noc_max1d_smooth(np.arange(2, 51), real_foc)
    else:
        noc_idx = get_noc(np.arange(2, 51), real_foc, plot_fit=True, plot_asm=False, fit_summary=True,
                          ttest_cutoff=False, iqr=False, ax=ax, use_median=False, spread_asm=True, spread_foc=True,
                          ci=ci, n_sigma=n_sigma)

    est = np.arange(2, 51)[noc_idx]
    interval_width = 5
    if simdata_results_path is not None:
        est, interval_width = correct_estimate(est, simdata_results_path)


    bottom, top = ax.get_ylim()
    ax.add_patch(mpatches.Rectangle((est-interval_width, bottom), 2*interval_width, top-bottom, fill=True,
                 facecolor='Grey', lw=0, alpha=0.25))

    ax.axvline(est, color='Grey', ls='--', zorder=0)

    yticks = np.array([1, 2, 3, 4], dtype=np.int)
    ax.set_yticks(yticks)
    ret = ax.set_yticklabels(yticks*100)
    xticks = np.array([10, 20, 30, 40, 50])
    ax.set_xticks(xticks)
    ret = ax.set_xticklabels(xticks)

    ax.set_ylabel("Prediction Accuracy (% of chance)", fontsize=fontsize)
    ax.set_xlabel("# Outcome clusters", fontsize=fontsize)
    return est

def plot_real_accuracy_chance_results(path, ax=None, fontsize='x-large'):
    ax = ax or plt.gca()
    real_noc, real_foc, real_accuracy, real_chance = read_clustering_results(path)

    real_data_est = get_real_noc(np.arange(2, 51), real_foc,
                                 smooth=True, use_median=False,
                                 spread_asm=True, spread_foc=True)


    x_noc, lower, med, upper = flatten_summarize(np.arange(2, 51),
                                                 real_chance, smooth=False,
                                                 iqr=True)

    plot_line(x_noc, med, lower=lower, upper=upper, ax=ax, color='gray')

    x_noc, lower, med, upper = flatten_summarize(np.arange(2, 51),
                                                 real_accuracy, smooth=False,
                                                 iqr=True)

    plot_line(x_noc, med, lower=lower, upper=upper, ax=ax, color='black')

    yticks = np.array([2, 4, 6, 8, 10], dtype=np.int)
    ax.set_yticks(yticks/10)
    ret = ax.set_yticklabels(yticks*10)
    xticks = np.array([10, 20, 30, 40, 50])
    ax.set_xticks(xticks)
    ret = ax.set_xticklabels(xticks)

    ax.set_ylabel("Prediction Accuracy (% of chance)", fontsize=fontsize)
    ax.set_xlabel("# Outcome clusters", fontsize=fontsize)