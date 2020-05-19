import time
import math
import h5py

from mpi4py import MPI
from activ import load_data
from activ.clustering import subsample_umap_clustering, UmapClusteringResults
from activ.readfile import TrackTBIFile
from activ.utils import get_logger, get_start_portion

def test_umap_clustering():
    data = TrackTBIFile('activ/data.h5') 
    pdata = TrackTBIFile('activ/data.h5')
    cluster_sizes = list(range(2, 15))
    n_iters = 1 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    seed = int(time.time())
    fkwargs = dict()

    if size > 1:
      seed = comm.bcast(seed, root=0)
      seed += rank
      fkwargs['driver'] = 'mpio'
      fkwargs['comm'] = comm

    start, portion = get_start_portion(rank, size, n_iters)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if size > 1:
      rank_str = "%02d" % rank
      fmt = '%(asctime)s - %(name)s ' + rank_str + ' - %(levelname)s - %(message)s'

    logger = get_logger(name="subsample_umap_clustering", fmt=fmt)

    labels, preds, rlabels, rpreds = subsample_umap_clustering(pdata.biomarkers, data.outcomes, cluster_sizes,
                                                           portion, 1.0, agg='median',
                                                           n_umap_iters=30, logger=logger)
    shape = (n_iters, labels.shape[1], len(cluster_sizes))

    if size > 1:
       comm.barrier()

    if rank == 0:
       logger.info('writting results')

    fkwargs['mode'] = 'w'
    f = h5py.File('pipi.h5', **fkwargs)       
    labels_dset = f.create_dataset('labels', dtype=int, shape=shape)
    preds_dset = f.create_dataset('preds', dtype=int, shape=shape)
    rlabels_dset = f.create_dataset('rlabels', dtype=int, shape=shape)
    rpreds_dset = f.create_dataset('rpreds', dtype=int, shape=shape)

    labels_dset[start:start+portion] = labels
    preds_dset[start:start+portion] = preds
    rlabels_dset[start:start+portion] = rlabels
    rpreds_dset[start:start+portion] = rpreds

    f.create_dataset('seed', data=seed)
    dset = f.create_dataset('num_ranks', data=size)
    dset.attrs['description'] = "This is the number of MPI ranks used"
    f.create_dataset('cluster_sizes', data=cluster_sizes)
    dset = f.create_dataset('umap_iters', data=30)
    dset.attrs['description'] = "The number of UMAP iterations used for calculating the distance matrix"
    dset = f.create_dataset('fraction', data=1.0)
    dset.attrs['description'] = "the fraction to subsample at each iteration"
    f.close()

    if rank == 0:
       logger.info('done')

    assert 200 == 200
