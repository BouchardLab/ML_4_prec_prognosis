"""
A subpackage for carrying out parametric bootstrap simulation of mixture model data
"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification


def simdata(n_bm_features, n_oc_features, n_blobs, n_samples, random_state=None, error_X=True):
    rs = check_random_state(random_state)

    kwargs = dict(n_samples=n_samples,
                  n_features=n_bm_features,
                  random_state=rs)
    kwargs['n_clusters_per_class'] = 1
    kwargs['n_classes'] = n_blobs
    kwargs['n_informative'] = int(n_blobs/2)+1
    kwargs['class_sep'] = 4
    X, labels = make_classification(**kwargs)

    beta = rs.normal(0, 1, (n_bm_features, n_oc_features))
    Y = X @ beta
    Y += rs.normal(0, 1, Y.shape)
    if error_X:
        X += rs.normal(0, 1, X.shape)
    return X, Y, labels

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
        'enet': MultiTaskElasticNetCV(),
        'random': None
    }

    desc = "simulate clusters using make_classification"

    def_cluster_sizes = [5, 10, 15, 20, 25, 30]
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='the directory to save output to')
    parser.add_argument('-c', '--cluster_sizes', type=int_list, help='number of clusters', default=def_cluster_sizes)
    parser.add_argument('-s', '--seed', type=check_seed, help='random seed. default is based on clock', default='')

    args = parser.parse_args()

    logger = get_logger(name='pbsim')
    logger.info(f'using seed {args.seed}')

    random_state = check_random_state(args.seed)


    dat = load_data()
    bm = dat.biomarkers
    oc = dat.outcomes

    try:
        f = h5py.File(args.outfile, 'a')
        f.attrs['seed'] = args.seed

        for i, n_clusters in enumerate(args.cluster_sizes):
            logger.info(f'simulating data for {n_clusters} clusters')
            gname = 'n_clusters-%02d' % n_clusters
            g = f.create_group(gname)
            g.attrs['n_clusters'] = n_clusters
            X, Y, labels = simdata(bm.shape[1], oc.shape[1], n_clusters, bm.shape[0], random_state=random_state)
            TrackTBIFile.write(g, X, Y)
            g.create_dataset('labels', data=labels)
    finally:
        f.close()
