import os.path as _op
import h5py as _h5py
import numpy as _np
import abc as _abc
import scipy.spatial.distance as _spd
import scipy.cluster.hierarchy as _sch
import logging as _logging
from time import time as _time

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree

import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score, cross_val_predict

from umap import UMAP
from .data_normalization import data_normalization
from .sampler import JackknifeSampler, BootstrapSampler, SubSampler
from .utils import check_random_state

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



def get_rank_size(comm):
    if comm is None:
        return 1, 1
    else:
        return comm.Get_rank(), comm.Get_size()

def compute_umap_distance(X, n_components, metric='euclidean', n_iters=30, agg='median', umap_kwargs=dict(), random_state=None):
    if agg  not in ('mean', 'median'):
        raise ValueError("Unrecognized argument for agg: '%s'" % agg)
    rand = check_random_state(random_state)
    n = X.shape[0]
    samples = np.zeros(((n*(n-1))//2, n_iters))
    umap = UMAP(n_components=n_components, random_state=rand, **umap_kwargs)
    for i in range(n_iters):
        emb = umap.fit_transform(X)
        samples[:, i] = pdist(emb, metric=metric)
    ret = None
    if agg == 'median':
        ret = np.median(samples, axis=1)
    elif agg == 'mean':
        ret = np.mean(samples, axis=1)
    return ret


def log(logger, msg):
    if logger is not None:
        logger.info(msg)

def _umap_cluster(X, y, umap_dims, rand, umap_kwargs, n_umap_iters, cluster_sizes, logger, classifier, cv, agg):
    """
    Run a UMAP clustering sweep

    Arguments:
        X - shape (n_samples, n_features)
            the predictor matrix to predict cluster labels with

        y - shape (n_samples, n_response_features)
            the response matrix to build cluster labels from


    Returns:
        labels - shape (n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate

        rand_labels - shape (n_samples, n_cluster_sizes)
            randomized labels computed for each sample of each boostrap replicate

        chances - shape (n_samples, n_cluster_sizes)
            predictions of randomized labels for each sample of each boostrap replicate
    """
    dist = compute_umap_distance(y, n_components=umap_dims, random_state=rand, umap_kwargs=umap_kwargs, n_iters=n_umap_iters, agg=agg)
    true_labels = cut_tree(linkage(dist, method='ward'), n_clusters=cluster_sizes)
    rand_labels = np.zeros(true_labels.shape, dtype=true_labels.dtype)
    preds = np.zeros(true_labels.shape, dtype=true_labels.dtype)
    chances = np.zeros(true_labels.shape, dtype=true_labels.dtype)
    for nclust in range(len(cluster_sizes)):
        log(logger, 'predicting labels from %d clusters' % cluster_sizes[nclust])
        rand_labels[:, nclust] = rand.permutation(true_labels[:, nclust])
        preds[:, nclust] = cross_val_predict(classifier, X, true_labels[:, nclust], cv=cv, n_jobs=1)
        chances[:, nclust] = cross_val_predict(classifier, X, rand_labels[:, nclust], cv=cv, n_jobs=1)
    return true_labels, rand_labels, preds, chances

def _run_umap_clustering(X, y, cluster_sizes, sampler, agg='median', metric='euclidean',
                        classifier=None, cv=5, n_umap_iters=30, umap_dims=2,
                        random_state=None, umap_kwargs=dict(), logger=None):

    """
    Arguments:

        X - shape (n_samples, n_features)
            the predictor matrix to predict cluster labels with

        y - shape (n_samples, n_response_features)
            the response matrix to build cluster labels from

        cluster_sizes - shape (n_cluster_sizes,)
            the cluster sizes to do predictions for

        sampler - object
            the resampler to use to generate samples from the original datasets

        classifier - object, default : RandomForestClassifier(n_estimators=100).
            the classifier to predict cluster labels with.

        cv - int or cross-validation generator, default: 5
            the number of CV folds or the cross-validation generator to use for
            predicting labels. If an integer is supplied, StratifiedKFold will be done

        n_umap_iters - int
            the number of iterations to use for calculating the UMAP

        random_state - int or RandomState
            the seed or RandomState to use

    Returns:

        labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate

        rand_labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            randomized labels computed for each sample of each boostrap replicate

        chances - shape (n_bootstraps, n_samples, n_cluster_sizes)
            predictions of randomized labels for each sample of each boostrap replicate
    """
    rand = check_random_state(random_state)
    n = X.shape[0]

    n_iters = sampler.get_n_iters(X)

    n_samples = sampler.get_n_samples(X)

    if classifier is None:
        classifier = RFC(100, random_state=rand)

    true_labels = np.zeros((n_iters, n_samples, len(cluster_sizes)), dtype=np.int32)
    rand_labels = true_labels.copy()
    preds = np.zeros(true_labels.shape, dtype=np.float64)
    chances = preds.copy()
    mask = np.ones(n, dtype=bool)

    uckwargs = dict(
        umap_dims=umap_dims,
        rand=rand,
        umap_kwargs=umap_kwargs,
        n_umap_iters=n_umap_iters,
        cluster_sizes=cluster_sizes,
        logger=logger,
        cv=cv,
        classifier=classifier,
	agg=agg
    )

    for i, (X_p, y_p) in enumerate(sampler.sample(X, y)):
        uckwargs['y'] = y_p
        uckwargs['X'] = X_p
        log(logger, 'computing UMAP distance matrix')
        true_labels[i], rand_labels[i], preds[i], chances[i] = _umap_cluster(**uckwargs)
    return true_labels, preds, rand_labels, chances


def jackknifed_umap_clustering(X, y, cluster_sizes, indices=None, agg='median', metric='euclidean',
                                 classifier=None, cv=5, n_umap_iters=30, umap_dims=2,
                                 random_state=None, umap_kwargs=dict(), logger=None):
    """
    Arguments:

        X - shape (n_samples, n_features)
            the predictor matrix to predict cluster labels with

        y - shape (n_samples, n_response_features)
            the response matrix to build cluster labels from

        indices - int
            the indices to do jackknife replicates for

        classifier - object, default : randomforestclassifier(n_estimators=100).
            the classifier to predict cluster labels with.

        cv - int or cross-validation generator, default: 5
            the number of cv folds or the cross-validation generator to use for
            predicting labels. if an integer is supplied, stratifiedkfold will be done

        n_umap_iters - int
            the number of iterations to use for calculating the umap

        random_state - int or randomstate
            the seed or randomstate to use

    Returns:

        labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate

        rand_labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            randomized labels computed for each sample of each boostrap replicate

        chances - shape (n_bootstraps, n_samples, n_cluster_sizes)
            predictions of randomized labels for each sample of each boostrap replicate
    """


    kwargs = dict(
        X=X,
        y=y,
        cluster_sizes=cluster_sizes,
        metric=metric,
        classifier=classifier,
        cv=cv,
        n_umap_iters=n_umap_iters,
        umap_dims=umap_dims,
        random_state=random_state,
        umap_kwargs=umap_kwargs,
        logger=logger,
	agg=agg
    )
    kwargs['sampler'] = JackknifeSampler(indices=indices)
    return _run_umap_clustering(**kwargs)


def bootstrapped_umap_clustering(X, y, cluster_sizes, n_iters, agg='median', metric='euclidean',
                              classifier=None, cv=5, n_umap_iters=30, umap_dims=2,
                              random_state=None, umap_kwargs=dict(), logger=None):

    """
    Arguments:

        X - shape (n_samples, n_features)
            the predictor matrix to predict cluster labels with

        y - shape (n_samples, n_response_features)
            the response matrix to build cluster labels from

        n_bootstraps - int
            the number of bootstraps to do

        classifier - object, default : randomforestclassifier(n_estimators=100).
            the classifier to predict cluster labels with.

        cv - int or cross-validation generator, default: 5
            the number of cv folds or the cross-validation generator to use for
            predicting labels. if an integer is supplied, stratifiedkfold will be done

        n_umap_iters - int
            the number of iterations to use for calculating the umap

        random_state - int or randomstate
            the seed or randomstate to use

    Returns:

        labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate

        rand_labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            randomized labels computed for each sample of each boostrap replicate

        chances - shape (n_bootstraps, n_samples, n_cluster_sizes)
            predictions of randomized labels for each sample of each boostrap replicate
    """

    kwargs = dict(
        X=X,
        y=y,
        cluster_sizes=cluster_sizes,
        metric=metric,
        classifier=classifier,
        cv=cv,
        n_umap_iters=n_umap_iters,
        umap_dims=umap_dims,
        random_state=random_state,
        umap_kwargs=umap_kwargs,
        logger=logger,
	agg=agg
    )
    kwargs['sampler'] = BootstrapSampler(n_iters, random_state=random_state)
    return _run_umap_clustering(**kwargs)


def subsample_umap_clustering(X, y, cluster_sizes, n_iters, subsample_size, agg='median', metric='euclidean',
                             classifier=None, cv=5, n_umap_iters=30, umap_dims=2,
                             random_state=None, umap_kwargs=dict(), logger=None):
    """
    Arguments:

        X - shape (n_samples, n_features)
            the predictor matrix to predict cluster labels with

        y - shape (n_samples, n_response_features)
            the response matrix to build cluster labels from

        n_iters - int
            the number of iterations to do

        subsample_size - int or float
            the size of the subsample to take

        classifier - object, default : randomforestclassifier(n_estimators=100).
            the classifier to predict cluster labels with.

        cv - int or cross-validation generator, default: 5
            the number of cv folds or the cross-validation generator to use for
            predicting labels. if an integer is supplied, stratifiedkfold will be done

        n_umap_iters - int
            the number of iterations to use for calculating the umap

        random_state - int or randomstate
            the seed or randomstate to use

    Returns:

        labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the labels computed for each sample of each boostrap replicate

        preds - shape (n_bootstraps, n_samples, n_cluster_sizes)
            the predictions for each sample of each boostrap replicate

        rand_labels - shape (n_bootstraps, n_samples, n_cluster_sizes)
            randomized labels computed for each sample of each boostrap replicate

        chances - shape (n_bootstraps, n_samples, n_cluster_sizes)
            predictions of randomized labels for each sample of each boostrap replicate
    """

    kwargs = dict(
        X=X,
        y=y,
        cluster_sizes=cluster_sizes,
        metric=metric,
        classifier=classifier,
        cv=cv,
        n_umap_iters=n_umap_iters,
        umap_dims=umap_dims,
        random_state=random_state,
        umap_kwargs=umap_kwargs,
        logger=logger,
	agg=agg
    )
    kwargs['sampler'] = SubSampler(n_iters, subsample_size=subsample_size, random_state=random_state)
    return _run_umap_clustering(**kwargs)

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


def read_data(path):
    f = _h5py.File(path, mode='r')
    try:
        labels = f['labels'][:]
        preds = f['preds'][:]
        rlabels = f['rlabels'][:]
        rpreds = f['rpreds'][:]
        cluster_sizes = f['cluster_sizes'][:]
    finally:
        f.close()

    n_subsamples = labels.shape[0]
    n_cluster_sizes = labels.shape[2]
    accuracy = np.zeros((n_cluster_sizes, n_subsamples))
    chance = np.zeros((n_cluster_sizes, n_subsamples))

    for sample in range(n_subsamples):
        for cl in range(n_cluster_sizes):
            sl = np.s_[sample, :, cl]
            accuracy[cl, sample] = accuracy_score(labels[sl], preds[sl])
            chance[cl, sample] = accuracy_score(labels[sl], rpreds[sl])
    foc = accuracy/chance

    return cluster_sizes, foc, accuracy, chance


def plot_line(x, med, lower=None, upper=None, ax=None, color='red', label=None, xlabel=None, ylabel=None, title=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    eb_kwargs = dict(x=x, y=med, color=color, fmt='-o', label=label)
    if lower is not None and upper is not None:
        eb_kwargs['yerr'] = [med-lower,upper-med]
    ax.errorbar(**eb_kwargs)
    if title is not None:
        ax.set_title(title, fontsize=20)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def flatten(noc, foc, filter_inf=True):
    x_train = np.repeat(noc, foc.shape[1])
    y_train = np.ravel(foc)
    if filter_inf:
        good_pts = np.where(np.logical_not(np.isinf(y_train)))[0]
        x_train = x_train[good_pts]
        y_train = y_train[good_pts]
    return x_train, y_train


def filter_iqr(noc, foc, s=1.5):
    mask = np.zeros(foc.size, dtype=bool)
    lower = np.zeros(noc.shape[0])
    med = np.zeros(noc.shape[0])
    upper = np.zeros(noc.shape[0])
    step = foc.shape[1]
    b = 0
    for i, c in enumerate(noc):
        lower[i], upper[i] = np.percentile(foc[i], [25, 75])
        e = step*(i+1)
        mask[b:e] = np.logical_and(foc[i] >= lower[i]*(1-s), foc[i] <= upper[i]*(1+s))
        b = e
    x_train, y_train = flatten(noc, foc, filter_inf=False)
    x_train = x_train[mask]
    y_train = y_train[mask]
    return x_train, y_train


def summarize_flattened(x, y, iqr=False):
    uniq = np.unique(x)
    lower = np.zeros(uniq.shape[0])
    med = np.zeros(uniq.shape[0])
    upper = np.zeros(uniq.shape[0])
    q = 1.96
    for i, c in enumerate(uniq):
        idx = x == c
        if iqr:
            lower[i], med[i], upper[i] = np.percentile(y[idx], [25, 50, 75])
        else:
            med[i] = np.mean(y[idx])
            sd = np.std(y[idx])
            lower[i] = med[i] - sd
            upper[i] = med[i] + sd
    return uniq, lower, med, upper


def linefit_func(x, a, b, c):
    return a + b * np.exp(c * x)

def ttest_min(noc_filt, foc_filt, mean, std, nobs, pvalue_cutoff):
    uniq = np.unique(noc_filt)
    pvalue = np.zeros(uniq.shape[0])

    for i, c in enumerate(uniq):
        idx = noc_filt == c
        _foc = foc_filt[idx]
        result = sps.ttest_ind_from_stats(mean1=np.mean(_foc),
                                          std1=np.std(_foc),
                                          nobs1=_foc.shape[0],
                                          mean2=mean,
                                          std2=std,
                                          nobs2=nobs,
                                          equal_var=False)
        _, pvalue[i] = result

    min_noc = np.where(pvalue > pvalue_cutoff)[0]
    if min_noc.shape[0] > 0:
        min_noc = min_noc[0]
    else:
        min_noc = -1
    return min_noc


def ci_overlap_min(noc_filt, foc_filt, mean, std, n_sigma=np.abs(sps.norm.ppf(0.025)), use_median=False, spread_asm=False, spread_foc=True):
    uniq = np.unique(noc_filt)

    asm_upper = mean
    asm_lower = mean

    if spread_asm:
        asm_std = n_sigma*std
        asm_upper += asm_std
        asm_lower -= asm_std

    if use_median and spread_foc:
        raise ValueError("Cannot use spread FOC using median")
    if use_median:
        print("using median")

    min_noc = -1
    for i, c in enumerate(uniq):
        idx = noc_filt == c
        _foc = foc_filt[idx]
        if use_median:
            foc_mean = np.median(_foc)
            foc_upper = foc_mean
            foc_lower = foc_mean
        else:
            foc_mean = np.mean(_foc)
            foc_upper = foc_mean
            foc_lower = foc_mean
            if spread_foc:
                foc_std = n_sigma*np.std(_foc)
                foc_upper += foc_std
                foc_lower -= foc_std

        if asm_upper > foc_lower and foc_upper > asm_lower:
            min_noc = i
            break

    return min_noc

def get_noc(noc, foc, pvalue_cutoff=0.05, fit_summary=True, plot=False, ttest_cutoff=True, ax=None, f_kwargs=None, a_kwargs=None, n_sigma=None, ci=0.95, use_median=False, spread_asm=False, spread_foc=True):
    # clean up and summarize data
    noc_filt, foc_filt = filter_iqr(noc, foc)
    fit_obs = noc_filt.shape[0]

    x_fit, y_fit = noc_filt, foc_filt
    lower, med, upper = None, None, None
    if fit_summary:
        _, lower, med, upper = summarize_flattened(noc_filt, foc_filt, iqr=False)
        fit_obs = lower.shape[0]
        x_fit, y_fit = _, med

    popt, pcov = spo.curve_fit(linefit_func, x_fit, y_fit, p0=np.array([-1, -1, 0]))

    lim = popt[0]
    lim_sd = np.sqrt(pcov[0,0])

    if n_sigma is None:
        n_sigma = np.abs(sps.norm.ppf((1-ci)/2))

    if ttest_cutoff:
        min_noc = ttest_min(noc_filt, foc_filt, lim, lim_sd, x_fit.shape[0], pvalue_cutoff)
    else:
        min_noc = ci_overlap_min(noc_filt, foc_filt, lim, lim_sd, n_sigma=n_sigma, spread_asm=spread_asm, spread_foc=spread_foc, use_median=use_median)


    if plot:
        if med is None:
            _, lower, med, upper = summarize_flattened(noc_filt, foc_filt, iqr=False)

        plot_line(noc, med, lower, upper)
        if ax is None:
            ax = plt.gca()
        x_plot = np.linspace(np.min(noc_filt), np.max(noc_filt), 1000)
        y_plot = linefit_func(x_plot, *popt)

        _kwargs = dict(color='black', ls='--')
        if isinstance(f_kwargs, dict):
            _kwargs.update(f_kwargs)
        ax.plot(x_plot, y_plot, **_kwargs)

        _kwargs = dict(color='gray', label='limit of linefit: %.3f' % lim)
        if isinstance(a_kwargs, dict):
            _kwargs.update(a_kwargs)

        ax.plot(x_plot, np.repeat(lim, 1000), **_kwargs)
        q = 1.96
        ax.fill_between(x_plot, np.repeat(lim-n_sigma*lim_sd, 1000), np.repeat(lim+n_sigma*lim_sd, 1000),
                        color='lightgray', label="limit 95%% CI: %.3f" % lim_sd)

    return min_noc
