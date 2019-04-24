import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import h5py
import os
import scipy

def heatmap(data, row_labels, col_labels, ax=None,
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
    im = ax.imshow(data, **kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)

    ax.set_yticklabels(row_labels)
    ax.set_xlabel(None)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #        rotation_mode="anchor")

    return im, cbar

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
