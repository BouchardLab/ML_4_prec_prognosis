import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import itertools
import h5py
import os
import scipy
from sklearn.metrics import accuracy_score

import matplotlib.colors as mpc
import seaborn as sns

def heatmap(data, row_labels=None, col_labels=None, ax=None, show_cbar=True,
            cbar_kw={}, cbarlabel="", xlab=None, ylab=None,
            title=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
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
        ax = plt.gca()

    kwargs.setdefault('cmap', 'binary')
    # Plot the heatmap
    im = ax.imshow(data, aspect='auto', **kwargs)

    # Create colorbar
    cbar = None
    if show_cbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, cax=cax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    if row_labels is not None:
    # We want to show all ticks...

        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(row_labels)
        if xlab is not None:
            ax.set_xlabel(xlab)

    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(col_labels)
        if ylab is not None:
            ax.set_ylabel(ylab)

    if title is not None:
        ax.set_title(title)

    return im, cbar


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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def nmf_bases_heatmap(data, col_labels, sort=True, ax=None,
            cumsum_thresh=0.99,
            cbar_kw={}, cbarlabel="", xlab=None, ylab=None,
            title=None, **kwargs):
    """
    Create a heatmap from outcome factors.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the outcome
                     factors for the rows
        col_labels : A list or array of length M with the outcome
                     labels for the columns
    Optional arguments:
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
        ax = plt.gca()

    kwargs.setdefault('cmap', 'binary')
    row_labels = np.arange(data.shape[0]) + 1


    plot_data = np.array(data)


    xlabels = col_labels
    if sort==True:
        factor_order = np.argsort(np.max(plot_data, axis=1))[::-1]
        row_labels = row_labels[factor_order]
        plot_data = plot_data[factor_order]
        to_sort = list()
        for i, c in enumerate(plot_data.T):
            mi = np.argmax(c)
            to_sort.append((mi, -1*c[mi], i))
        new_order = [t[2] for t in sorted(to_sort)]
        plot_data = plot_data[:,new_order]
        xlabels = [xlabels[i] for i in new_order]

    for i in range(plot_data.shape[0]):
        order = np.argsort(plot_data[i])[::-1]
        cumsum = np.cumsum(plot_data[i,order])
        cumsum = cumsum / cumsum[-1]
        plot_data[i][order[cumsum > cumsum_thresh]] = 0.0

    # Plot the heatmap
    im = ax.imshow(plot_data, **kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_yticks(np.arange(plot_data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(xlabels)

    ax.set_yticklabels(row_labels)
    ax.set_xlabel(None)
    if xlab is not None:
        ax.set_xlabel(xlab,fontsize=36)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=36)
    if title is not None:
        ax.set_title(title, fontsize=48)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")

def nmf_heatmap(data, labels, numclusters):
    reordered = data[:,labels]
    heatmap(reordered, list(range(1,numclusters+1)), labels+1,xlab='NMF Factors', ylab='Number of Clusters', title='NMF Weights Heatmap')

def nmf_boxplot(data, labels):
    ax = plt.gca()
    ax.boxplot(data[:, labels], whis = [25, 75])
    ax.set_xlabel('NMF Factors')
    ax.set_ylabel('Weights across all Patients')
    ax.set_title('NMF Weights Boxplot')
    labels2 = labels+np.ones(len(labels), dtype=np.int)
    ax.set_xticklabels(labels2)

def outcomes_histogram(oc_data, oc_features, indices, nrows=1, ncols=1, figsize=(6,6)):
    fig, ax = plt.subplots(nrows,ncols,sharey=True,figsize=figsize)
    ax = ax.flatten()
    count = 0
    for ii, ind in enumerate(indices):
        data = oc_data[:,ind]
        name = oc_features[ind]
        ax[count].hist(data, color='black')
        ax[count].set_title('{}'.format(name))
        count += 1


def conf_alliter(size, predicted, actual, iterations):
    confalliter = []
    for j in range(iterations):
        conf = np.zeros((size,size))
        labs = list(map(int, actual[j][size-2]))
        preds = list(map(int, predicted[j][size-2]))
        for l, p in zip(labs, preds):
            conf[l,p] += 1
        confalliter.append(conf)
    return confalliter

def accuracy(mat):
    return mat.trace()/mat.sum()



def plot_clustering_results(input_file, plot_this, specific_plot_name=None):
    f = h5py.File(input_file, 'r')
    predictions = np.squeeze(f['score'][:], axis=1)
    norm_predictions = np.squeeze(f['norm_score'][:], axis=1)
    cluster_sizes = f['cluster_sizes'][:]
    labels = np.squeeze(f['clusters'][:], axis=1)
    num_iter = predictions.shape[0]
    num_clust = len(cluster_sizes)
    fold_over_chance = np.zeros((num_clust, num_iter))
    raw_acc = np.zeros((num_clust, num_iter))
    normalized_acc = np.zeros((num_clust, num_iter))
    for jj,c in enumerate(cluster_sizes):
        all50mat = conf_alliter(c, predictions, labels, num_iter)
        norm_all50mat = conf_alliter(c, norm_predictions, labels, num_iter)
        acc = [accuracy(i) for i in all50mat]
        norm_acc = [accuracy(i) for i in norm_all50mat]
        foc = [float(x/y) for x, y in zip(acc, norm_acc)]
        fold_over_chance[jj,:] = foc
        raw_acc[jj,:] = acc
        normalized_acc[jj,:] = norm_acc
    plt.rcParams["figure.figsize"] = [10,10]
    cmap = plt.get_cmap("tab10")
    lower = np.asarray([np.percentile(fold_over_chance[i,:], 25) for i in range(num_clust)])
    upper = np.asarray([np.percentile(fold_over_chance[i,:], 75) for i in range(num_clust)])
    med = np.median(fold_over_chance, axis=1)
    raw_med = np.median(raw_acc, axis=1)
    raw_lower = [np.percentile(raw_acc[i,:], 25) for i in range(num_clust)]
    raw_upper = [np.percentile(raw_acc[i,:], 75) for i in range(num_clust)]
    norm_med = np.median(normalized_acc, axis=1)
    norm_lower = [np.percentile(normalized_acc[i,:], 25) for i in range(num_clust)]
    norm_upper = [np.percentile(normalized_acc[i,:], 75) for i in range(num_clust)]
    yerr = upper-lower
    rel_iqr = yerr/med

    _, tail = os.path.split(input_file)
    if specific_plot_name is not None:
        plot_title = specific_plot_name
    else:
        plot_title = tail.split('.')[0]
    save_name = plot_title + '_foc.pdf'
    save_name = os.path.join('/Users/ahyeon/Desktop/activ misc/', save_name)
    if plot_this == 'foc':
        plt.errorbar(cluster_sizes, med, yerr=[med-lower,upper-med], color = 'red',fmt='-o', label='Fold over chance')
        plt.title(str(plot_title), fontsize=20)
        plt.xlabel("Cluster sizes", fontsize=20)
        plt.ylabel("Fold over chance", fontsize=20)
    elif plot_this == 'raw_and_chance':
        plt.errorbar(cluster_sizes, raw_med, yerr=[raw_med-raw_lower,raw_upper-raw_med], color='black',fmt='-o', label='Raw accuracy')
        plt.errorbar(cluster_sizes, norm_med, yerr=[norm_med-norm_lower, norm_upper-norm_med], color='grey',fmt='-o', label='Chance accuracy')
        plt.title(str(plot_title), fontsize=20)
        plt.xlabel("Cluster sizes", fontsize=20)
        plt.ylabel("Raw and Chance Accuracy", fontsize=20)
        plt.legend()
    else:
        plt.plot(cluster_sizes, rel_iqr, '-o', color='black')
        plt.title(str(plot_title), fontsize=20)
        plt.xlabel("Cluster sizes", fontsize=20)
        plt.ylabel("Normalized IQR", fontsize=20)
    plt.savefig(save_name)

def plot_clustering_results_activ_data(group_name, plot_this, specific_plot_name=None):
    with h5py.File('/Users/ahyeon/data/activ/activ_data.h5','r') as hf:
        data = hf['umap_clustering/{}'.format(group_name)]
        f = data
        predictions = np.squeeze(f['score'][:], axis=1)
        norm_predictions = np.squeeze(f['norm_score'][:], axis=1)
        cluster_sizes = f['cluster_sizes'][:]
        labels = np.squeeze(f['clusters'][:], axis=1)
        num_iter = predictions.shape[0]
        num_clust = len(cluster_sizes)
        fold_over_chance = np.zeros((num_clust, num_iter))
        raw_acc = np.zeros((num_clust, num_iter))
        normalized_acc = np.zeros((num_clust, num_iter))
        for jj,c in enumerate(cluster_sizes):
            all50mat = conf_alliter(c, predictions, labels, num_iter)
            norm_all50mat = conf_alliter(c, norm_predictions, labels, num_iter)
            acc = [accuracy(i) for i in all50mat]
            norm_acc = [accuracy(i) for i in norm_all50mat]
            foc = [float(x/y) for x, y in zip(acc, norm_acc)]
            fold_over_chance[jj,:] = foc
            raw_acc[jj,:] = acc
            normalized_acc[jj,:] = norm_acc
        plt.rcParams["figure.figsize"] = [10,10]
        cmap = plt.get_cmap("tab10")
        lower = np.asarray([np.percentile(fold_over_chance[i,:], 25) for i in range(num_clust)])
        upper = np.asarray([np.percentile(fold_over_chance[i,:], 75) for i in range(num_clust)])
        med = np.median(fold_over_chance, axis=1)
        raw_med = np.median(raw_acc, axis=1)
        raw_lower = [np.percentile(raw_acc[i,:], 25) for i in range(num_clust)]
        raw_upper = [np.percentile(raw_acc[i,:], 75) for i in range(num_clust)]
        norm_med = np.median(normalized_acc, axis=1)
        norm_lower = [np.percentile(normalized_acc[i,:], 25) for i in range(num_clust)]
        norm_upper = [np.percentile(normalized_acc[i,:], 75) for i in range(num_clust)]
        yerr = upper-lower
        rel_iqr = yerr/med

        if specific_plot_name is not None:
            plot_title = specific_plot_name
        else:
            plot_title = group_name

        if plot_this == 'foc':
            plt.errorbar(cluster_sizes, med, yerr=[med-lower,upper-med], color = 'red',fmt='-o', label='Fold over chance')
            plt.title(str(plot_title), fontsize=20)
            plt.xlabel("Cluster sizes", fontsize=20)
            plt.ylabel("Fold over chance", fontsize=20)
        elif plot_this == 'raw_and_chance':
            plt.errorbar(cluster_sizes, raw_med, yerr=[raw_med-raw_lower,raw_upper-raw_med], color='black',fmt='-o', label='Raw accuracy')
            plt.errorbar(cluster_sizes, norm_med, yerr=[norm_med-norm_lower, norm_upper-norm_med], color='grey',fmt='-o', label='Chance accuracy')
            plt.title(str(plot_title), fontsize=20)
            plt.xlabel("Cluster sizes", fontsize=20)
            plt.ylabel("Raw and Chance Accuracy", fontsize=20)
            plt.legend()
        else:
            plt.plot(cluster_sizes, rel_iqr, '-o', color='black')
            plt.title(str(plot_title), fontsize=20)
            plt.xlabel("Cluster sizes", fontsize=20)
            plt.ylabel("Normalized IQR", fontsize=20)

def return_plot_items(path):
    f = h5py.File(path, mode='r')
    try:
        labels = f['labels'][:]
        preds = f['preds'][:]
        rlabels = f['rlabels'][:]
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

    accuracy_lower = np.percentile(accuracy, 25, axis=1)
    accuracy_med = np.percentile(accuracy, 50, axis=1)
    accuracy_upper = np.percentile(accuracy, 75, axis=1)

    chance_lower = np.percentile(chance, 25, axis=1)
    chance_med = np.percentile(chance, 50, axis=1)
    chance_upper = np.percentile(chance, 75, axis=1)

    foc_lower = np.percentile(foc, 25, axis=1)
    foc_med = np.percentile(foc, 50, axis=1)
    foc_upper = np.percentile(foc, 75, axis=1)

    iqr = foc_upper-foc_lower
    rel_iqr = iqr/np.median(foc, axis=1)
    return foc, iqr, rel_iqr, cluster_sizes

def cluster_plot(path, plot_this, title, ax=None):
    foc, iqr, rel_iqr, cluster_sizes = return_plot_items(path)
    if plot_this == 'foc':
        ax.errorbar(cluster_sizes, foc_med, yerr=[foc_med-foc_lower,foc_upper-foc_med], color = 'red',fmt='-o', label='Fold over chance')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Cluster sizes")
        ax.set_ylabel("Fold over chance")
    elif plot_this == 'iqr':
        ax.plot(cluster_sizes, iqr, '-o', color='black')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Cluster sizes")
        ax.set_ylabel("IQR")
    elif plot_this == 'rel_iqr':
        ax.plot(cluster_sizes, rel_iqr, '-o', color='gray')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Cluster sizes")
        ax.set_ylabel("Relative IQR")

