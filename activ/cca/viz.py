import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from ..viz import get_labels


def _check_array(v):
    if isinstance(v, pd.Series):
        v = v.values
    if len(v.shape) == 1:
        v = v.reshape(v.shape[0], 1)
    elif len(v.shape) == 2:
        if v.shape[1] != 1:
            raise ValueError('v must be 1D: shape = %s' % str(v.shape))
        v = v.reshape(v.shape[0])
    return v


def cross_decomp_scatter(x, y, regressor=LinearRegression(), labels=None, fitline=True,
                         solid_points=False, markeredgewidth=1, markersize=None,
                         fontsize=24, legend_fontsize=None,
                         title=None, xlabel=None, ylabel=None, legend_title=None, ax=None):
    """
    Args:
        x:                  the independent variable to plot
        y:                  the dependent variable to plot
        regressor:          the scikit-learn regressor object to use draw a line-fit
        labels:             category labels to apply
        solid_points:       plot solid points instead of circles
        markeredgewidth:    the edge widgth of markers
        markeredgesize:     the size of markers
        title:              the title to add to the plot
        xlabel:             the label to add to the X-axis
        ylabel:             the label to add to the Y-axis
        legend_title:       the title to give to the legend
        ax:                 the matplotlib Axes object draw the plot on
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    _x = _check_array(x)
    _y = _check_array(y)

    colors, patches = None, None
    mk_kwargs = dict(marker='o', color='none')
    scatter_kwargs = dict()

    if solid_points:
        markersize = markersize or 5
        scatter_kwargs['c'] = colors
        colors_key = 'c'
    else:
        markersize = markersize or 9
        mk_kwargs['markerfacecolor'] = 'none'
        mk_kwargs['markeredgewidth'] = markeredgewidth or 1
        colors_key = 'edgecolors'
        scatter_kwargs['facecolors'] = 'none'
        scatter_kwargs['linewidths'] = mk_kwargs['markeredgewidth']

    mk_kwargs = dict(marker='o', color='none')
    scatter_kwargs = dict()

    if solid_points:
        markersize = markersize or 5
        scatter_kwargs['c'] = colors
        colors_key = 'c'
    else:
        markersize = markersize or 9
        mk_kwargs['markerfacecolor'] = 'none'
        mk_kwargs['markeredgewidth'] = markeredgewidth or 1
        colors_key = 'edgecolors'
        scatter_kwargs['facecolors'] = 'none'
        scatter_kwargs['linewidths'] = mk_kwargs['markeredgewidth']

    mk_kwargs['markersize'] = markersize
    scatter_kwargs['s'] = markersize**2


    if labels is not None:
        colors, patches = get_labels(labels, solid_points=solid_points, marker_kwargs=mk_kwargs)
    else:
        colors = ['black'] * _x.shape[0]

    scatter_kwargs[colors_key] = colors

    ax.scatter(_x, _y, **scatter_kwargs)

    if legend_fontsize is None:
        legend_fontsize = fontsize-4

    if patches is not None:
        ax.legend(handles=patches, title=legend_title, loc=2, fontsize=legend_fontsize)

    ax.tick_params('both', labelsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if fitline:
        y_pred = cross_val_predict(regressor, _x, _y, cv=5)
        cv_r2 = r2_score(_y, y_pred)

        regressor.fit(_x, _y)
        # raw_r2 = r2_score(_y, regressor.predict(_x))

        xfit = np.linspace(min(_x),max(_x), 1000).reshape((1000,1))
        yfit = regressor.predict(xfit)

        ax.plot(xfit, yfit, color='black')
        x_pos, y_pos = (0.7, 0.1)
        ax.text(x_pos, y_pos, "$R^2$ = %0.2f" % cv_r2, size=fontsize, transform=ax.transAxes)
    return ax
