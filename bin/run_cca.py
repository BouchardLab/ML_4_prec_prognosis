import argparse
import pyuoi
from activ import readfile
from activ.cca.alscca import TALSCCA
from activ.utils import get_logger, check_seed, check_dir

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale

from scipy.stats import pearsonr

import warnings

import seaborn as sns

import matplotlib.colors as mpc


def combine(df, domain=True, subdomain=True):
    if domain and subdomain:
        if len(df['sub-domain']) > 0:
            subdd = df['domain'] + ' - ' + df['sub-domain']
        else:
            subdd = df['domain']
    elif subdomain:
        subdd = df['sub-domain']
    else:
        subdd = df['domain']
    return subdd


def get_domain_weight(df, ddict, coef, domain=True, subdomain=True, scale=False):
    """

    Args:
        scale: scale weights by the number of variables in the domain
    """
    mask = coef != 0
    weights = coef[mask]
    weights = np.abs(weights)
    weights /= weights.sum()
    col = df.columns[mask]
    subdd = ddict.filter(items=col, axis=0)
    subdd = combine(subdd, domain, subdomain)
    dweight = dict()
    for d in np.unique(subdd):
        mask = subdd == d
        if scale:
            dweight[d] = weights[mask].mean()
        else:
            dweight[d] = weights[mask].sum()
    s = pd.Series(data = dweight)
    s /= s.sum()
    return s


def multi_stemplot(values, varclass=None, ax=None, labels=None, palette='Set1', luts=None):
    """
    Build stemplot with variables colored by class
    """
    if isinstance(values, np.ndarray):
        if len(values.shape) == 1:
            values = values.reshape(1, -1)
    if ax is None:
        ax = plt.gca()

    n_plots = len(values)

    maxstem = np.max([np.max(np.abs(values[i])) for i in range(n_plots)])
    shift = maxstem*2.1
    offsets = shift * np.arange(n_plots)

    offset = 0.0
    for i in range(n_plots):
        classes = varclass[i]
        offset = offsets[i]
        order = np.argsort(classes)
        values[i] = values[i][order]
        classes = classes[order]

        for c in np.unique(classes):
            val = np.zeros(len(classes)) + offset
            val[classes==c] += values[i][classes==c]
            col = mpc.to_hex(luts[i][c])
            ax.stem(val, use_line_collection=True, basefmt=col, linefmt=col, markerfmt=' ', bottom=offset)
        ax.stem(np.zeros(len(classes))+offset, basefmt='black', linefmt='black', markerfmt=' ', use_line_collection=True, bottom=offset)

    pos = offsets
    ax.get_xaxis().set_ticks([])
    yaxis = ax.get_yaxis()
    yaxis.set_ticks(pos)
    ticklabels = np.arange(n_plots)
    if labels is not None:
        ticklabels = labels
    yaxis.set_ticklabels(ticklabels, fontsize=20)


def calculate_parameters(X, Y, gcs_simple, n_components=6):
    """
    Run cross-validation to compute the best parameter set
    """
    random_state = check_random_state(args.seed)
    grid = ParameterGrid({'max_iters':[1000], 'n_components': [n_components], 'alpha_x': 10.0**(-1*np.arange(1, 5)), 'alpha_y': 10.0**(-1*np.arange(1,5))})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings(record=False):
        warnings.simplefilter('ignore')
        scores = np.zeros((cv.n_splits, len(grid)))
        for i, (train, test) in enumerate(cv.split(X, gcs_simple)):
            X_train, Y_train = X[train], Y[train]
            X_test, Y_test = X[test], Y[test]
            for j, params in enumerate(grid):
                talscca = TALSCCA(**params).fit(X_train, Y_train)
                X_cca, Y_cca = talscca.transform(X_test, Y_test)
                scores[i, j] = pearsonr(X_cca.flatten(), Y_cca.flatten())[0]
    return grid[np.argmax(scores.mean(axis=0))]


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input TrackTBIFile')
parser.add_argument('-o', '--output', type=check_dir, help='the file to output to')
parser.add_argument('-s', '--seed', type=check_seed, help='the seed to use for running the pipeline', default='')
parser.add_argument('-c', '--n_components', type=int, help='the number of canonical variates to compute', default=6)
parser.add_argument('-p', '--parameters', type=json.loads, help='a JSON script with the CCA parameters', default=None)

args = parser.parse_args()

data = readfile(args.input)

X = scale(data.biomarkers)
Y = scale(data.outcomes)

if parameters is None:
    gcs_simple = X[:, np.where(data.biomarker_features == 'GCSMildModSevereRecode')[0][0]]
    args.parameters = calculate_parameters(X, Y, gcs_simple)

params = args.parameters.copy()
params['max_iters'] = 4000
params['n_components'] = 6
params['random_state'] = args.seed
talscca = TALSCCA(**params)
cv_X, cv_Y = talscca.fit_transform(X, Y)

def add_dataset(g, name, ar):
    dset = g.require_dataset(name, dtype=ar.dtype, shape=ar.shape)
    dset[:] = ar

with h5py.File(args.input, 'a') as f:
    g = f.require_group('cca')
    add_dataset(g, 'X_variates', cv_X)
    add_dataset(g, 'Y_variates', cv_Y)
    add_dataset(g, 'X_loadings', talscca.X_components_)
    add_dataset(g, 'Y_loadings', talscca.Y_components_)
    for k, v in params.items():
        g.attrs[k] = v


