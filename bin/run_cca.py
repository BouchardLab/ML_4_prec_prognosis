import argparse
import json
from activ import TrackTBIFile
from activ.cca.alscca import TALSCCA
from activ.utils import get_logger, check_seed

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale

from scipy.stats import pearsonr

import warnings


def calculate_parameters(tbifile, n_components=6, max_iters=1000):
    """
    Run cross-validation to compute the best parameter set
    """
    X = tbifile.biomarkers
    Y = tbifile.outcomes
    random_state = check_random_state(args.seed)
    grid = ParameterGrid({'max_iters':[max_iters], 'n_components': [n_components], 'alpha_x': 10.0**(-1*np.arange(1, 5)), 'alpha_y': 10.0**(-1*np.arange(1,5))})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings(record=False):
        warnings.simplefilter('ignore')
        scores = np.zeros((cv.n_splits, len(grid)))
        for i, (train, test) in enumerate(cv.split(X, tbifile.gcs_simple)):
            X_train, Y_train = X[train], Y[train]
            X_test, Y_test = X[test], Y[test]
            for j, params in enumerate(grid):
                talscca = TALSCCA(random_state=random_state, **params).fit(X_train, Y_train)
                X_cca, Y_cca = talscca.transform(X_test, Y_test)
                scores[i, j] = pearsonr(X_cca.flatten(), Y_cca.flatten())[0]
    return grid[np.argmax(scores.mean(axis=0))]


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input TrackTBIFile')
parser.add_argument('-s', '--seed', type=check_seed, help='the seed to use for running the pipeline', default='')
parser.add_argument('-c', '--n_components', type=int, help='the number of canonical variates to compute', default=6)
parser.add_argument('-i', '--max_iters', type=int, help='the maximum number of iterators to use', default=1000)
parser.add_argument('-B', '--bm_alpha', type=int, help='regularization to use when regressing outcomes onto biomarkers', default=0.1)
parser.add_argument('-O', '--oc_alpha', type=int, help='regularization to use when regressing biomarkers onto outcomes', default=0.1)

grp = parser.add_mutually_exclusive_group()
grp.add_argument('--cv', action='store_true', default=False, help='run CV to determine best alpha parameters')
grp.add_argument('-p', '--parameters', type=json.loads, help='a JSON script with additional CCA parameters', default=None)

args = parser.parse_args()

logger = get_logger()
tbifile = TrackTBIFile(args.input)
tbifile.biomarkers = scale(tbifile.biomarkers)
tbifile.outcomes = scale(tbifile.outcomes)

tals_args = dict(
    n_components=args.n_components,
    alpha_x=args.bm_alpha,
    alpha_y=args.oc_alpha,
    max_iters=args.max_iters
)

if isinstance(args.parameters, dict):
    logger.info(f'using parameters {args.parameters}')
    tals_args.update(args.parameters)
elif args.cv:
    logger.info('determing best regularization parameters by cross-validation')
    params = calculate_parameters(tbifile, n_components=args.n_components, max_iters=args.max_iters)
    logger.info(f'best parameters: {params}')
    tals_args.update(params)

logger.info('running TALS-CCA')
cca = TALSCCA(**tals_args, random_state=args.seed)
bm_cv, oc_cv = cca.fit_transform(tbifile.biomarkers, tbifile.outcomes)
bm_ld, oc_ld = cca.X_components_, cca.Y_components_

metadata = {'cca_params': str(TALSCCA)}

logger.info('writing results')
TrackTBIFile.write_cca(args.input, bm_cv, oc_cv, bm_ld, oc_ld, metadata=metadata, overwrite=True)
