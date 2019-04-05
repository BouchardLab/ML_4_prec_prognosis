import os.path as _op
import h5py as _h5py
import numpy as _np
import scipy.spatial.distance as _spd
import scipy.cluster.hierarchy as _sch
import logging as _logging
from time import time as _time

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score, cross_val_predict

from umap import UMAP
from .data_normalization import data_normalization

def path_tuple(type_name, **kwargs):
    from collections import namedtuple
    keys = sorted(kwargs.keys())
    tup = namedtuple(type_name, keys)
    return tup(**kwargs)

class UmapClusteringResults(object):

    path = path_tuple("UmapClusteringResultsPath",
        scores = 'score',
        norm_scores = 'norm_score',
        cluster_sizes = 'cluster_sizes',
        umap_dimensions = 'umap_dimensions',
        umap_n_neighbors = "umap_n_neighbors",
        umap_min_dist = "umap_min_dist",
        seed = "seed",
        clusters = "clusters",
        embeddings = "umap_embeddings"
    )

    def __init__(self, results_path):
        self.outdir = _op.dirname(_op.abspath(results_path))
        f = _h5py.File(results_path, 'r')
        self.acc_samples = f[self.path.scores][:]
        self.chance_samples = f[self.path.norm_scores][:]
        self.cluster_sizes = f[self.path.cluster_sizes][:]
        self.umap_dims = f[self.path.umap_dimensions][:]
        self.clusters = f[self.path.clusters][:]
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


# completely rewrite this to do what Kris says
#   for nb in n_bootstraps
#       distmat = median([pdist(UMAP().fit_transform()) for nu in n_umap_iters])
#       dendro = linkage(distmat)
#       for c in n_clusters:
#           labels = cut_tree(dendro)
#           pred = cross_val_predict(classifier, predict_data, labels, cv=cv_folds)
#           score = accuracy_score(label, pred)


def check_random_state(random_state):
    if random_state is None:
        rand = np.random
    elif isinstance(random_state, (np.int32, np.int64, np.int16, np.int8, int)):
        rand = np.random.RandomState(random_state)
    else:
        rand = random_state
    return rand

def compute_umap_distance(X, n_components, metric='euclidean', n_iters=30, agg='median', umap_kwargs=dict(), random_state=None):
    if agg  not in ('mean', 'median'):
        raise ValueError("Unrecognized argument for agg: '%s'" % agg)
    n = X.shape[0]
    samples = np.zeros(((n*(n-1))/2, n_iters))
    umap = UMAP(n_components=n_components, metric=metric, random_state=random_state, **umap_kwargs)
    for i in range(n_iters):
        samples[:, i] = pdist(umap.fit_transform(X), metric=metric)
    ret = None
    if agg == 'median':
        ret = np.median(samples, axis=1)
    elif agg == 'mean':
        ret = np.mean(samples, axis=1)
    return ret

def bootstrapped_umap_clustering(X, y, n_bootstraps, cluster_sizes, metric='euclidean',
                                 n_umap_iters=30, umap_dims=2, random_state=None, umap_kwargs=dict()):

    """
    Returns:
        labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate
    """
    rand = check_random_state(random_state)
    n = y.shape[0]
    it = range(n_bootstraps)
    labels = np.zeros((n_bootstraps, n, len(cluster_sizes)))
    preds = np.zeros(labels.shape)
    for i in it:
        indices = rand.randint(n, size=n)
        y_p = y[indices]
        X_p = X[indices]
        dist = compute_umap_distance(y_p, n_components=umap_dims, random_state=rand, umap_kwargs=umap_kwargs)
        labels[i] = cut_tree(linkage(dist, method='ward'), n_clusters=cluster_sizes)
        for nclust in range(labels.shape[1]):
            preds[i, :, nclust] = cross_val_pred(classifier, X_p, labels[i, :, nclust])
    return labels, preds

def umap_cluster_sweep(n_iters, cluster_data, cluster_sizes, umap_dims=None, metric='mahalanobis',
                       predict_data=None, h5group=None, classifier=RFC(100),
                       precomputed_embeddings=None, single_dim=False, collapse=False,
                       umap_args=None, cv_folds=5, mpicomm=None,
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
        precomputed_embeddings      : UMAP embeddings that have already been computed
        single_dim                  : use single UMAP dimension
        collapse                    : use the mean distance matrix across UMAP embeddings for clustering.
                                      Note: This will change the shape of outputs

    Use 5-fold stratified cross-validation for scoring.

    If h5group is supplied, the following datasets will be written to it:
        score               - a dataset with shape (iterations, len(umap_dims), len(cluster_sizes), n)
                              that stores predict scores
        norm_score          - a dataset with shape (iterations, len(umap_dims), len(cluster_sizes), n)
                              that stores random guess predict scores
        clusters            - the cluster assignments for each iteration and UMAP dimension
        seed                - the value used to seed the random number generator
        umap_dimensions     - the UMAP dimensions that were tested
        cluster_sizes       - the cluster sizes that were tested
        umap_n_neighbors    - the UMAP n_neighbors parameter
        umap_min_dist       - the UMAP min_dist parameter

    """
    if seed is None:
        seed = int(round(_time() * 1000) % 2**32)
    iterations = range(n_iters)
    rank = 0
    if mpicomm:
        rank = mpicomm.Get_rank()
        iterations = range(rank, n_iters, mpicomm.Get_size())
        if rank != 0:
            seed = None
        seed = mpicomm.bcast(seed, root=0)
        seed += rank
    _np.random.seed(seed)

    if logger is None:
        logger = _logging.getLogger('umap_cluster_sweep')

    def log1(msg):
        if rank == 0:
            logger.info(msg)

    if predict_data is None:
        log1('using cluster_data predict clusters, which will be inferred from cluster_data')
        predict_data = cluster_data
    if predict_data.shape[0] != cluster_data.shape[0]:
        raise ValueError("predict_data and cluster_data must have the same number of samples")
    n = cluster_data.shape[0]
    log1('predicting clusters with %s' % str(classifier))
    log1('UMAP dimensions: %s' % str(umap_dims))
    log1('Cluster sizes: %s' % str(cluster_sizes))
    if umap_args is None:
        umap_args = dict()

    n_neighbors = umap_args.get('n_neighbors', 10)
    min_dist = umap_args.get('min_dist', 0.000)
    n_samples = cluster_data.shape[0]

    if collapse:
        output_shape = (n_iters, len(cluster_sizes), n)
        umap_params_shape = (n_iters, len(umap_dims))
        clusters_shape = (n_iters, len(cluster_sizes), n_samples)
    else:
        output_shape = (n_iters, len(umap_dims), len(cluster_sizes), n)
        umap_params_shape = (n_iters, len(umap_dims))
        clusters_shape = (n_iters, len(umap_dims), len(cluster_sizes), n_samples)
    embeddings_shape = (n_iters, n_samples, sum(umap_dims))
    emb_scale = _np.zeros(embeddings_shape[2], dtype=int)
    idx = 0
    for d in umap_dims:
        for i in range(d):
            emb_scale[idx] = d
            idx += 1

    close_grp = False
    if h5group is not None:
        # write to a file path if that's what we were given
        if isinstance(h5group, str):
            if mpicomm is not None:
                h5group = _h5py.File(h5group, 'w', driver='mpio', comm=mpicomm)
            else:
                h5group = _h5py.File(h5group, 'w')
            close_grp = True
        logger.info('saving results to %s (%s)' % (h5group.name, h5group.file.filename))
        score = h5group.create_dataset(UmapClusteringResults.path.scores, shape=output_shape, dtype=_np.float64)
        norm_score = h5group.create_dataset(UmapClusteringResults.path.norm_scores, shape=output_shape, dtype=_np.float64)
        clusters = h5group.create_dataset(UmapClusteringResults.path.clusters, shape=clusters_shape, dtype=int)
        all_embeddings = h5group.create_dataset(UmapClusteringResults.path.embeddings, shape=embeddings_shape, dtype=_np.float64)

        emb_scale_dset = h5group.create_dataset(UmapClusteringResults.path.embeddings + '_dimscale', dtype=int, shape=(len(emb_scale),))
        all_embeddings.dims.create_scale(emb_scale_dset, "Embedding size")
        all_embeddings.dims[2].attach_scale(emb_scale_dset)
        umap_dims_dset = h5group.create_dataset(UmapClusteringResults.path.umap_dimensions, dtype=int, shape=(len(umap_dims),))
        n_neighbors_dset = h5group.create_dataset(UmapClusteringResults.path.umap_n_neighbors, dtype=int, shape=())
        min_dist_dset = h5group.create_dataset(UmapClusteringResults.path.umap_min_dist, dtype=_np.float64, shape=())
        cluster_sizes_dset = h5group.create_dataset(UmapClusteringResults.path.cluster_sizes, dtype=int, shape=(len(cluster_sizes),))
        seed_dset = h5group.create_dataset(UmapClusteringResults.path.seed, dtype=int, shape=())
        if rank == 0:
            emb_scale_dset[()] = emb_scale
            umap_dims_dset[()] = umap_dims
            n_neighbors_dset[()] = n_neighbors
            min_dist_dset[()] = min_dist
            cluster_sizes_dset[()] = cluster_sizes
            seed_dset[()] = seed
    else:
        all_embeddings = _np.zeros(embeddings_shape, dtype=_np.float64)
        score = _np.zeros(output_shape)
        norm_score = _np.zeros(output_shape)
        clusters = _np.zeros(clusters_shape, dtype=int)

    result = _np.zeros(output_shape[1:], dtype=_np.float64)
    norm_result = _np.zeros(output_shape[1:])
    all_clusters = _np.zeros(clusters_shape[1:], dtype=int)
    dists = _np.zeros((len(umap_dims), n*(n-1)//2), dtype=_np.float64)
    normalized = data_normalization(cluster_data, 'z-score')
    for iter_i in iterations:
        logger.info("BEGIN iteration %s" % iter_i)
        embeddings = None
        if precomputed_embeddings is None:
            dim_b = 0
            for ii, num_dims in enumerate(umap_dims): # umap dimension
                dim_e = dim_b + num_dims
                embedding = UMAP(n_components=num_dims, n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(normalized)
                all_embeddings[iter_i, :, dim_b:dim_e] = embedding
                dim_b = dim_e
            embeddings = all_embeddings
        else:
            embeddings = precomputed_embeddings
        for i, d in enumerate(umap_dims):
            dists[i] = _spd.pdist(embeddings[iter_i, :, emb_scale == d], metric=metric)
        if collapse:
            dist = dists.mean(axis=0)
            cluster_results = _sch.cut_tree(_sch.linkage(dist, method='ward'), cluster_sizes)
            for jj in range(cluster_results.shape[1]):
                labels = cluster_results[:, jj]
                num_clusters = cluster_sizes[jj]
                logger.info('num_clusters: %s' % num_clusters)
                result[jj] = cross_val_predict(classifier, predict_data, labels, cv=5)
                norm_result[jj] = cross_val_predict(classifier, predict_data, _np.random.permutation(labels), cv=5)
            all_clusters[:]  = cluster_results.T
        else:
            for ii in range(dists.shape[0]):
                dist = dists[ii]
                cluster_results = _sch.cut_tree(_sch.linkage(dist, method='ward'), cluster_sizes)
                for jj in range(cluster_results.shape[1]):
                    labels = cluster_results[:, jj]
                    num_clusters = cluster_sizes[jj]
                    logger.info('umap_dims: %s, num_clusters: %s' % (umap_dims[ii], num_clusters))
                    result[ii,jj] = cross_val_predict(classifier, predict_data, labels, cv=5)
                    norm_result[ii,jj] = cross_val_predict(classifier, predict_data, _np.random.permutation(labels), cv=5)
                all_clusters[ii] = cluster_results.T

        logger.info("END iteration %s" % iter_i)

        # write everything back for return
        score[iter_i] = result
        norm_score[iter_i] = norm_result
        clusters[iter_i] = all_clusters

        # zero out temporary variables - this is unnecessary, but do it for good measure
        result.fill(0.0)
        norm_result.fill(0.0)
        all_clusters.fill(0)
        dists.fill(0)

    log1("finished call to umap_cluster_sweep")
    if close_grp:
        logger.info("closing %s" % h5group.filename)
        h5group.close()

    return score, norm_score, seed, all_embeddings, clusters
