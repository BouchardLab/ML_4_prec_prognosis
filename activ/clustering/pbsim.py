"""
A subpackage for carrying out parametric bootstrap simulation of mixture model data
"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.covariance import oas


def sim_clusters(centers, covs, n_samples=100, random_state=None):
    """
    Simulate multivariate Gaussian clusters using custom centers and covariances

    Args:
        centers (array-like)            : centers of Gaussians
                                          shape: (n_clusters, n_features)
        covs (array-like)               : covariance of Gaussians.
                                          shape: (n_clusters, n_features, n_features)
                                                 or (n_features, n_features)
        n_samples (array-like)          : number of samples in each cluster
                                          shape: scalar or (n_clusters,)

        random_state                    : the random state to use
    """
    random_state = check_random_state(random_state)
    if np.issubdtype(type(n_samples), np.integer):
        n_samples = [n_samples] * centers.shape[0]
    if len(covs.shape) == 2:
        covs = np.tile(covs.ravel(), centers.shape[0])\
               .reshape((centers.shape[0], covs.shape[0], covs.shape[1]))
    samples = list()
    labels = list()
    for i, (mean, cov, n) in enumerate(zip(centers, covs, n_samples)):
        samples.append(random_state.multivariate_normal(mean, cov, n))
        labels.append([i]*n)
    ret = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels)
    return ret, labels


def transform_multivariate_data(X, lm, res=None, random_state=None):
    """
    Transform multivariate data according the fit of a linear model, and
    add error.

    Args:
        X (array-like)                  : the data to transform
        lm (sklearn linear model)       : a fitted linear model
        res (array-like)                : residual error terms to compute an emperical
                                          error distribution from. If None, use standard
                                          normal error
        random_state                    : the random state to use
    """
    random_state = check_random_state(random_state)
    Y = X @ lm.coef_.T + lm.intercept_
    if res is None:
        res_means = np.zeros(Y.shape[1])
        res_var = np.diag(np.ones(Y.shape[1]))
    else:
        res_means = np.mean(res, axis=0)
        res_var = np.diag(np.var(res, axis=0))
    Y += random_state.multivariate_normal(res_means, res_var, X.shape[0])
    return Y


def get_covs(X, labels, cov_est=lambda x: oas(x)[0]):
    """
    Get covariance matrices for each cluster

    Args:
        X (array-like)                  : the data to get coviariances for
        labels (array-like)             : the cluster labels for each data point
        cov_est (function)              : a function that takes a (n_samples, n_features) shaped
                                          array, and returns an (n_features, n_features) matrix
    """
    uniq_lbls = np.sort(np.unique(labels))
    ret = np.zeros((uniq_lbls.shape[0], X.shape[1], X.shape[1]))
    for i, lbl in enumerate(uniq_lbls):
        ret[i] = cov_est(X[labels==lbl])
    return ret


def get_means(X, labels):
    """
    Get centers for each cluster

    Args:
        X (array-like)                  : the data to get means for
        labels (array-like)             : the cluster labels for each data point
    """
    uniq_lbls = np.sort(np.unique(labels))
    ret = np.zeros((uniq_lbls.shape[0], X.shape[1]))
    for i, lbl in enumerate(uniq_lbls):
        ret[i] = np.mean(X[labels==lbl], axis=0)
    return ret


def get_n_samples(labels):
    """
    Get the number of samples for each cluster

    Args:
        labels (array-like)             : the cluster labels for each data point
    """
    uniq_lbls = np.sort(np.unique(labels))
    return np.array([np.sum(labels == _) for _ in uniq_lbls])

if __name__ == '__main__':
    import argparse
    from ..readfile import load_data, TrackTBIFile
    from ..utils import check_seed, int_list, get_logger
    from activ.clustering import compute_umap_distance
    from scipy.cluster.hierarchy import linkage, cut_tree
    import h5py
    from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, MultiTaskElasticNetCV
    from sklearn.ensemble import RandomForestRegressor

    REG_CHOICES = {
        'rf': RandomForestRegressor(n_estimators=200),
        'linear': LinearRegression(),
        'lasso': MultiTaskLassoCV(),
        'enet': MultiTaskElasticNetCV()
    }

    desc = "simulate clusters using parameters calculated from original TBI data"

    def_cluster_sizes = [5, 10, 15, 20, 25, 30]
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='the directory to save output to')
    parser.add_argument('-c', '--cluster_sizes', type=int_list, help='number of clusters', default=def_cluster_sizes)
    parser.add_argument('-s', '--seed', type=check_seed, help='random seed. default is based on clock', default='')
    parser.add_argument('-r', '--regressor', choices=REG_CHOICES.keys(), default='linear',
                        help='the type of regressor to use when transforming clusters')

    args = parser.parse_args()

    logger = get_logger(name='pbsim')
    logger.info(f'using seed {args.seed}')

    random_state = check_random_state(args.seed)

    dat = load_data()
    bm = dat.biomarkers
    oc = dat.outcomes

    # cluster data
    logger.info('computing iterative UMAP distance matrix')
    dist = compute_umap_distance(bm, 2)
    logger.info('running HCL and cutting to get clusters of size %s' % ", ".join(map(str, args.cluster_sizes)))
    all_labels = cut_tree(linkage(dist, method='ward'), n_clusters=args.cluster_sizes)

    # compute linear relationship between biomarkers and outcomes
    logger.info('computing linear relationship and residuals')
    regressor = REG_CHOICES[args.regressor]
    logger.info('Regressing outcomes onto biomarkers with\n %s' % str(regressor))
    regressor.fit(bm, oc)
    # compute residuals, we will use this to estimate an error distrubtion
    res = oc - regressor.predict(bm)

    def simdata(__labels):
        # compute multivariate Gaussian parameters from data
        empcov = get_covs(bm, __labels)
        mean = get_means(bm, __labels)
        n_samples = get_n_samples(__labels)
        # simulate data with these parameters
        X, labels = sim_clusters(mean, empcov, n_samples, random_state)
        # transform simulated data with computed linear relationship
        Y = transform_multivariate_data(X, regressor, res, random_state)
        return X, Y, labels

    try:
        f = h5py.File(args.outfile, 'w')
        f.attrs['seed'] = args.seed
        if hasattr(regressor, 'coef_'):
            f.create_dataset('beta', data=regressor.coef_)
            f.create_dataset('intercept', data=regressor.intercept_)

        for i, n_clusters in enumerate(args.cluster_sizes):
            logger.info(f'simulating data for {n_clusters} clusters')
            gname = 'n_clusters-%02d' % n_clusters
            g = f.create_group(gname)
            g.attrs['n_clusters'] = n_clusters
            X, Y, labels = simdata(all_labels[:, i])
            TrackTBIFile.write(g, X, Y)
            g.create_dataset('labels', data=labels)
    finally:
        f.close()
