import h5py

from activ.clustering import umap_cluster_sweep, UmapClusteringResults
from activ.readfile import TrackTBIFile


def test_umap_cluster_sweep():
    n_iters = 1
    data = TrackTBIFile('activ/data.h5')
    outcomes = data.outcomes
    umap_dims = []
    cluster_sizes = list(range(2, 15))
    h5group = h5py.File('umap_cluster_sweep.h5', mode='w')
    umap_cluster_sweep(n_iters,
                       outcomes,
                       cluster_sizes,
                       umap_dims,
                       'mahalanobis',
                       None,
                       h5group)
    h5group.close()
    results = UmapClusteringResults('umap_cluster_sweep.h5')
    results.make_heatmap()


def test_umap_cluster_sweep_with_no_h5group():
    n_iters = 1
    data = TrackTBIFile('activ/data.h5')
    outcomes = data.outcomes
    umap_dims = []
    cluster_sizes = list(range(2, 15))
    umap_cluster_sweep(n_iters,
                       outcomes,
                       cluster_sizes,
                       umap_dims,
                       'mahalanobis',
                       None)
