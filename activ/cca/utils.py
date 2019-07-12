import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from ..viz import get_labels


def cross_decomp_scatter(x, y, regressor=LinearRegression(), labels=None, fitline=True,
                         solid_points=False,
                         title=None, xlabel=None, ylabel=None, legend_title=None, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    _x = x
    _y = y
    if len(x.shape) == 1:
        _x = x.reshape(x.shape[0], 1)
    if len(y.shape) == 2:
        if y.shape[1] != 1:
            raise ValueError('y must be 1D: shape = %s' % str(y.shape))
        _y = y.reshape(y.shape[0])

    colors, patches = None, None
    if labels is not None:
        colors, patches = get_labels(labels)
    else:
        colors = ['black'] * _x.shape[0]

    if solid_points:
        ax.scatter(_x, _y, c=colors, s=5)
    else:
        ax.scatter(_x, _y, edgecolors=colors, facecolors='none', s=80)

    if patches is not None:
        ax.legend(handles=patches, title=legend_title, loc=2)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if fitline:
        y_pred = cross_val_predict(regressor, _x, _y, cv=5)
        cv_r2 = r2_score(_y, y_pred)

        regressor.fit(_x, _y)
        raw_r2 = r2_score(_y, regressor.predict(_x))

        xfit = np.linspace(min(_x),max(_x), 1000).reshape((1000,1))
        yfit = regressor.predict(xfit)

        ax.plot(xfit, yfit, color='black')
        ax.text(0.7, 0.1, "$R^2$ (fit) = %0.4f\n$R^2$ (cv) = %0.4f" % (raw_r2, cv_r2), size=16, transform=ax.transAxes)
    return ax
