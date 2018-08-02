import os.path as _op
import h5py as _h5py
import numpy as _np
import logging as _logging
from time import time as _time

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

from activ.pipeline import run_umap, cluster_range

class UmapClusteringResults(object):

    def __init__(self, results_path):
        self.outdir = _op.dirname(_op.abspath(results_path))
        f = _h5py.File(results_path, 'r')
        self.acc_samples = f['score'][:]
        self.chance_samples = f['norm_score'][:]
        self.cluster_sizes = f['cluster_sizes'][:]
        self.umap_dims = f['umap_dimensions'][:]
        self.__acc = self.reshape(self.acc_samples)
        self.__chance = self.reshape(self.chance_samples)
        f.close()

    def reshape(self, dat):
        s = dat.shape
        x = _np.zeros((s[1], s[2], s[0]*s[3]))
        for i in range(s[1]):
            for j in range(s[2]):
                b = 0
                for k in range(s[0]):
                    e = b + s[3]
                    x[i,j,b:e] = dat[k,i,j]
                    b = e
        return x

    def adjust(self, op='mean'):
        if not hasattr(_np, op):
            msg = "unrecoginized operation for ndarray: '%s'" % op
            raise ValueError(msg)
        func = getattr(_np, op)
        acc = func(self.__acc, axis=2)
        chance = func(self.__chance, axis=2)
        return acc/chance

    def make_heatmap(self, path=None, op='mean'):
        """
        Make a heatmap. Save the heatmap to the specified path.
        If *path* is not specified, save to *heatmap.png* in the
        same directory as the given results file.

        Args:
            path        : the path to save the heatmap to
            op          : the summary operation. options are 'median'
                          or 'mean'. default is 'mean'
        """
        import matplotlib.pyplot as _plt
        from activ.analytics import heatmap
        dest = path
        if dest is None:
            dest = _op.join(self.outdir, 'heatmap_%s.png' % op)
        title = 'accuracy averaged across\n%s iterations of %s-fold CV' % \
                (self.acc_samples.shape[0], self.acc_samples.shape[3])
        if op == 'mean':
            title = "Mean " + title
        elif op == 'median':
            title = "Median " + title
        else:
            raise ValueError("unrecoginized operation: use 'median' or 'mean'")
        adjusted = self.adjust(op=op)
        im, cbar = heatmap(adjusted, self.umap_dims, self.cluster_sizes,
                           xlab="Number of Clusters",
                           ylab="Number of UMAP dimensions",
                           title=title)
        _plt.savefig(dest)
        return im, cbar

def umap_cluster_sweep(iterations, cluster_data, umap_dims, cluster_sizes,
                       predict_data=None, h5group=None, classifier=RFC(100),
                       umap_args=None, cv_folds=5,
                       seed=None, logger=None):
    """
    Run UMAP-HCL parameter sweep.

    Arguments:
        iterations                  : the number of iterations to run
        cluster_data                : the data to transform and cluster
        umap_dims                   : a list of the number of UMAP dimensions to test
        cluster_sizes               : a list of the cluster sizes to test
        predict_data                : the data to predict cluster assignments with
        h5group                     : the HDF5 group to write results to
        classifier                  : the classifier to use for predicting clusters from predict_data
        umap_args                   : a dict of additional arguments for running UMAP
        cv_folds                    : the number of CV folds to use
        logger                      : a logger for logging progress to
        seed                        : the seed to use for random number generation. default is to use
                                      time

    Use 5-fold stratified cross-validation for scoring.

    If h5group is supplied, the following datasets will be written to it:
        score               - a dataset with shape (iterations, len(umap_dims), len(cluster_sizes), 5)
                              that stores scores
        norm_score          - a dataset with shape (iterations, len(umap_dims), len(cluster_sizes), 5)
                              that stores random guess scores
        clusters            - the cluster assignments for each iteration and UMAP dimension
        seed                - the value used to seed the random number generator
        umap_dimensions     - the UMAP dimensions that were tested
        cluster_sizes       - the cluster sizes that were tested
        umap_n_neighbors    - the UMAP n_neighbors parameter
        umap_min_dist       - the UMAP min_dist parameter

    """
    if logger is None:
        logger = _logging.getLogger('umap_cluster_sweep')
    if predict_data is None:
        logger.info('using cluster_data predict clusters, which will be inferred from cluster_data')
        predict_data = cluster_data
    if predict_data.shape[0] != cluster_data.shape[0]:
        raise ValueError("predict_data and cluster_data must have the same number of samples")
    logger.info('predicting clusters with %s' % str(classifier))
    logger.info('UMAP dimensions: %s' % str(umap_dims))
    logger.info('Cluster sizes: %s' % str(cluster_sizes))
    if umap_args is None:
        umap_args = dict()

    n_iters = iterations
    n_neighbors = umap_args.get('n_neighbors', 10)
    min_dist = umap_args.get('min_dist', 0.000)
    n_samples = cluster_data.shape[0]

    output_shape = (n_iters, len(umap_dims), len(cluster_sizes), cv_folds)
    umap_params_shape = (n_iters, len(umap_dims))
    clusters_shape = (n_iters, len(umap_dims), len(cluster_sizes), n_samples)

    if seed is None:
        seed = int(round(_time() * 1000) % 2**32)
    _np.random.seed(seed)

    close_grp = False
    if h5group is not None:
        # write to a file path if that's what we were given
        if isinstance(h5group, str):
            h5group = _h5py.File(h5group, 'w')
            close_grp = True
        logger.debug('saving results to %s (%s)' % (h5group.name, h5group.file.filename))
        score = h5group.create_dataset("score", shape=output_shape, dtype=_np.float64)
        norm_score = h5group.create_dataset("norm_score", shape=output_shape, dtype=_np.float64)
        clusters = h5group.create_dataset("clusters", shape=clusters_shape, dtype=int)
        h5group.create_dataset("umap_dimensions", data=umap_dims)
        h5group.create_dataset("umap_n_neighbors", data=n_neighbors)
        h5group.create_dataset("umap_min_dist", data=min_dist)
        h5group.create_dataset("cluster_sizes", data=cluster_sizes)
        h5group.create_dataset("seed", data=seed)
    else:
        score = _np.zeros(output_shape)
        norm_score = _np.zeros(output_shape)
        clusters = _np.zeros(clusters_shape, dtype=int)

    result = _np.zeros(output_shape[1:], dtype=_np.float64)
    norm_result = _np.zeros(output_shape[1:])
    all_clusters = _np.zeros(clusters_shape[1:], dtype=int)
    for iter_i in range(n_iters):
        logger.info("== begin iteration %s ==" % iter_i)
        for ii, num_dims in enumerate(umap_dims): # umap dimension
            embedding = run_umap(cluster_data,  num_dims,
                                 n_neighbors=n_neighbors,
                                 min_dist=min_dist)
            cluster_results = cluster_range(embedding, cluster_sizes, method='ward')
            for jj in range(cluster_results.shape[1]):
                labels = cluster_results[:, jj]
                num_clusters = cluster_sizes[jj]
                logger.info('umap dimensions: %s, num cluster_sizes: %s' % (num_dims, num_clusters))
                result[ii,jj] = cross_val_score(classifier, predict_data, labels, cv=5)
                norm_result[ii,jj] = cross_val_score(classifier, predict_data, _np.random.permutation(labels), cv=5)
            all_clusters[ii] = cluster_results.T
        logger.info("== end iteration %s ==" % iter_i)

        # write everything back for return
        score[iter_i] = result
        norm_score[iter_i] = norm_result
        clusters[iter_i] = all_clusters

        # zero out temporary variables - this is unnecessary, but do it for good measure
        result.fill(0.0)
        norm_result.fill(0.0)
        all_clusters.fill(0)

    if close_grp:
        h5group.close()

    return score, norm_score, seed, clusters
