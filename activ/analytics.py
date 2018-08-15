import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

