import os
import numpy as np
import h5py
import logging
import sys
from time import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

from activ.pipeline import run_umap, cluster_range
from activ import load_data

def umap_pipeline(iteration, cluster_data, umap_dims, cluster_sizes,
                  predict_data=None, h5group=None, classifier=RFC(100),
                  logger=None):
    logger = logging.getLogger('umap_pipeline')
    if logger is None:
        logger = logging.getLogger('umap_pipeline')
    if predict_data is None:
        logger.info('using cluster_data predict clusters, which will be inferred from cluster_data')
        predict_data = cluster_data
    if predict_data.shape[0] != cluster_data.shape[0]:
        raise ValueError("predict_data and cluster_data must have the same number of samples")
    logger.info('predicting clusters with %s' % str(classifier))
    logger.info('UMAP dimensions: %s' % str(umap_dims))
    logger.info('Cluster sizes: %s' % str(cluster_sizes))

    cv_folds = 5
    n_iters = iteration
    n_neighbors = 10
    min_dist = 0.000
    n_samples = cluster_data.shape[0]

    output_shape = (n_iters, len(umap_dims), len(cluster_sizes), cv_folds)
    umap_params_shape = (n_iters, len(umap_dims))
    clusters_shape = (n_iters, len(umap_dims), len(cluster_sizes), n_samples)
    embdict = {}
    treedict = {}
    if h5group is not None:
        logger.debug('saving results to %s (%s)' % (h5group.name, h5group.file.filename))
        score = h5group.create_dataset("score", shape=output_shape, dtype=np.float64)
        norm_score = h5group.create_dataset("norm_score", shape=output_shape, dtype=np.float64)
        umap_seeds = h5group.create_dataset("umap_seeds", shape=umap_params_shape, dtype=int)
        clusters = h5group.create_dataset("clusters", shape=clusters_shape, dtype=int)
        h5group.create_dataset("umap_dimensions", data=umap_dims)
        h5group.create_dataset("umap_n_neighbors", data=n_neighbors)
        h5group.create_dataset("umap_min_dist", data=min_dist)
        h5group.create_dataset("cluster_sizes", data=cluster_sizes)
    else:
        score = np.zeros(output_shape)
        norm_score = np.zeros(output_shape)
        umap_seeds = np.zeros(umap_params_shape, dtype=int)
        clusters = np.zeros(clusters_shape, dtype=int)

    result = np.zeros(output_shape[1:], dtype=np.float64)
    norm_result = np.zeros(output_shape[1:])
    seeds = np.zeros(umap_params_shape[1:], dtype=int)
    all_clusters = np.zeros(clusters_shape[1:], dtype=int)
    for iter_i in range(n_iters):
        embeddings = []
        logger.info("== begin iteration %s ==" % iter_i)
        for ii, num_dims in enumerate(umap_dims): # umap dimension
            seed = int(round(time() * 1000) % 2**32)
            seeds[ii] = seed
            embedding = run_umap(cluster_data,  num_dims,
                                 n_neighbors=n_neighbors,
                                 min_dist=min_dist,
                                 random_state=seed)
            cluster_results = cluster_range(embedding, cluster_sizes, method='ward')
            for jj in range(cluster_results.shape[1]):
                labels = cluster_results[:, jj]
                num_clusters = cluster_sizes[jj]
                logger.info('umap dimensions: %s, num cluster_sizes: %s' % (num_dims, num_clusters))
                result[ii,jj] = cross_val_score(classifier, predict_data, labels, cv=5)
                norm_result[ii,jj] = cross_val_score(classifier, predict_data, np.random.permutation(labels), cv=5)
            all_clusters[ii] = cluster_results.T
        logger.info("== end iteration %s ==" % iter_i)

        # write everything back for return
        score[iter_i] = result
        norm_score[iter_i] = norm_result
        umap_seeds[iter_i] = seeds
        clusters[iter_i] = all_clusters

        # zero out temporary variables - this is unnecessary, but do it for good measure
        result.fill(0.0)
        norm_result.fill(0.0)
        seeds.fill(0)
        all_clusters.fill(0)
    return score, norm_score, umap_seeds, clusters




if __name__ == '__main__':
    def int_list(string):
        return list(map(int, string.split(",")))
    import argparse

    parser = argparse.ArgumentParser(usage="%(prog)s [options] output_h5",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))

    parser.add_argument("output_h5", type=str, help="the path to the HDF5 file to save results to")
    parser.add_argument("-i", "--iters", type=int, help="the number of iterations to run", default=50)
    parser.add_argument("-u", "--umap_dims", type=int_list, help="a comma-separated list of the UMAP dimensions",
                        default=list(range(2, 11)))
    parser.add_argument("-c", "--cluster_size", type=int_list, help="a comma-separated list of the cluster sizes",
                        default=list(range(2, 15)))
    parser.add_argument("-f", "--force", help="force rerunning i.e. overwrite output_h5", action='store_true', default=False)
    parser.add_argument("-q", "--quiet", help="make log messages quiet", action='store_true', default=False)

    pargs = parser.parse_args()

    args = list()
    kwargs = dict()
    data = load_data()

    args.append(pargs.iters)
    args.append(data.data_oc)
    args.append(pargs.umap_dims)
    args.append(pargs.cluster_size)
    kwargs['predict_data'] = data.data_bm

    logger = logging.getLogger('umap_pipeline')
    logger.addHandler(logging.StreamHandler(sys.stderr))
    log_level = logging.INFO
    if pargs.quiet:
        log_level = logging.WARNING
    logger.setLevel(log_level)
    kwargs['logger'] = logger

    path = pargs.output_h5
    if os.path.exists(path) and not pargs.force:
        sys.stderr.write("%s already exists. cautiously exiting. use -f to override\n" % path)
        sys.exit(255)
    f = h5py.File(path, 'w')
    kwargs['h5group'] = f
    start = datetime.now()
    logger.info("Begin UMAP pipeline at %s" % start.strftime('%Y-%m-%d %H:%M:%S'))
    res = umap_pipeline(*args, **kwargs)
    end = datetime.now()
    logger.info("End UMAP pipeline at %s" % end.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Time elapsed: %s" % str(end-start))
    f.close()
