import h5py
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.signal as signal

def read_clustering_results(path):
    """
    Read clustering results

    Returns:
        (cluster_sizes, fold-over-chance, accuracy, chance)
    """
    f = h5py.File(path, mode='r')
    try:
        labels = f['labels'][:]
        preds = f['preds'][:]
        rpreds = f['rpreds'][:]
        cluster_sizes = f['cluster_sizes'][:]
    finally:
        f.close()

    n_subsamples = labels.shape[0]
    n_cluster_sizes = labels.shape[2]
    accuracy = np.zeros((n_cluster_sizes, n_subsamples))
    chance = np.zeros((n_cluster_sizes, n_subsamples))

    for sample in range(n_subsamples):
        for cl in range(n_cluster_sizes):
            sl = np.s_[sample, :, cl]
            accuracy[cl, sample] = accuracy_score(labels[sl], preds[sl])
            chance[cl, sample] = accuracy_score(labels[sl], rpreds[sl])
    foc = accuracy/chance

    return cluster_sizes, foc, accuracy, chance


def get_avg(ar, i):

    p = float('nan')
    for ii in reversed(range(i)):
        if not np.isinf(ar[ii]):
            p = ar[ii]
            break
    n = float('nan')
    for ii in range(i+1, ar.shape[0]):
        if not np.isinf(ar[ii]):
            n = ar[ii]
            break
    ret = 0.0
    if not np.isnan(p):
        ret += p/2
    if not np.isnan(n):
        ret += n/2
    if ret == 0.0:
        raise ValueError('flanking values average to 0.0 -- '
                         'this seems unlikely -- '
                         'probably could not find finite values')
    return ret


def flatten(noc, foc, filter_inf=True, smooth=True):
    """
    Flatten fold-over-chance and number-of-clusters arrays into
    paired data points

    Args:
        noc          : the number of clusters (as stored in results file)
        foc          : fold-over-chance predictive accuracy (as stored in results file)

    Returns:
        (noc_flattened, foc_flattened)
    """
    if filter_inf:
        x_train = np.repeat(noc, foc.shape[1])
        y_train = np.ravel(foc)
        good_pts = np.where(np.logical_not(np.isinf(y_train)))[0]
        x_train = x_train[good_pts]
        y_train = y_train[good_pts]
    else:
        w = 11
        c = w // 2
        kernel = signal.hann(w)
        kernel /= kernel.sum()
        new_foc = np.copy(foc)
        for i in range(foc.shape[1]):
            sub = foc[:, i]
            inf_vals = np.where(np.isinf(sub))[0]
            for j in inf_vals:
                sub[j] = get_avg(sub, j)
            if smooth:
                new_foc[:, i] = np.convolve(foc[:, i], kernel)[c:-c]
        x_train = np.repeat(noc, new_foc.shape[1])
        y_train = np.ravel(new_foc)
    return x_train, y_train


def filter_iqr(noc, foc, s=1.5):
    """
    Remove data points outside the interquartile-range of fold-over-chance

    Args:
        noc          : the number of clusters (as stored in results file)
        foc          : fold-over-chance predictive accuracy (as stored in results file)
    """
    mask = np.zeros(foc.size, dtype=bool)
    lower = np.zeros(noc.shape[0])
    upper = np.zeros(noc.shape[0])
    step = foc.shape[1]
    b = 0
    for i, c in enumerate(noc):
        lower[i], upper[i] = np.percentile(foc[i], [25, 75])
        e = step*(i+1)
        mask[b:e] = np.logical_and(foc[i] >= lower[i]*(1-s), foc[i] <= upper[i]*(1+s))
        b = e
    x_train, y_train = flatten(noc, foc, filter_inf=False)
    x_train = x_train[mask]
    y_train = y_train[mask]
    return x_train, y_train


def summarize_flattened(x, y, iqr=False):
    """
    Remove data points outside the interquartile-range of fold-over-chance

    Args:
        noc          : the number of clusters (as stored in results file)
        foc          : fold-over-chance predictive accuracy (as stored in results file)
    """
    uniq = np.unique(x)
    lower = np.zeros(uniq.shape[0])
    med = np.zeros(uniq.shape[0])
    upper = np.zeros(uniq.shape[0])
    for i, c in enumerate(uniq):
        idx = x == c
        if iqr:
            lower[i], med[i], upper[i] = np.percentile(y[idx], [25, 50, 75])
        else:
            med[i] = np.mean(y[idx])
            sd = np.std(y[idx]) / med[i]
            lower[i] = med[i] - sd
            upper[i] = med[i] + sd
    return uniq, lower, med, upper


def plot_line(x, med, lower=None, upper=None, color='red', label=None, xlabel=None, ylabel=None, title=None, ax=None, fontsize=None, marker='-o'):
    """
    Plot value as a function of number of clusters, including error bars
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    yerr = None
    if lower is not None and upper is not None:
        yerr = [med - lower, upper - med]
    ax.errorbar(x, med, yerr=yerr, color=color, fmt=marker, label=label)
    if title is not None:
        ax.set_title(title, fontsize=20)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def flatten_summarize(noc, measure, filter_inf=True, smooth=True, iqr=True):
    """
    Flatten data and return summary metrics for plotting
    a measure as a function of number of clusters with error bars

    Returns:
        (x, lower_quartile, middle_quartile, upper_quartile)
    """
    x_flat, y_flat = flatten(noc, measure, filter_inf=filter_inf, smooth=smooth)
    return summarize_flattened(x_flat, y_flat, iqr=iqr)


def plot_results(noc, foc, error_bars=True, smooth=True, **plot_args):
    """
    Plot fold-over-chance with error bars

    Args:
        noc          : the number of clusters (as stored in results file)
        foc          : fold-over-chance predictive accuracy (as stored in results file)
    """
    noc_flat, foc_flat = flatten(noc, foc, smooth=smooth, filter_inf=False)
    x, lower, med, upper = summarize_flattened(noc_flat, foc_flat, iqr=True)
    kwargs = plot_args.copy()
    if error_bars:
        kwargs['lower'] = lower
        kwargs['upper'] = upper
    plot_line(x, med, **kwargs)
