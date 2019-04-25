from activ import load_data
import time
import math
from mpi4py import MPI
import h5py

from activ.clustering import subsample_umap_clustering, UmapClusteringResults
from activ.readfile import TrackTBIFile
from activ.utils import get_logger, get_start_portion

from argparse import ArgumentParser



def int_list(string):
    if ':' in string:
        ar = [int(a) for a in string.split(":")]
        return list(range(ar[0], ar[1]+1))
    else:
        return list(map(int, string.split(",")))

parser = ArgumentParser(usage="%(prog)s [options] output_h5")


parser.add_argument('output', type=str, help='the output HDF5 file')

parser.add_argument('-s', '--seed', type=int, help='the seed to use for running the pipeline', default=None)

parser.add_argument("-d", "--data", type=str, help="the Track TBI dataset file. use activ.load_data() by default", default=None)
parser.add_argument("-p", "--pdata", type=str, help="the Track TBI dataset file to use for predictions. use activ.load_data() by default", default=None)
parser.add_argument("-i", "--iterations", type=int, help="the number of subsampling iterations to run", default=50)
parser.add_argument("-f", "--fraction", type=float, help="the fraction of the original dataset to subsample", default=1.0)
parser.add_argument("-c", "--cluster_sizes", type=int_list, help="a comma-separated list of the cluster sizes",
                    default=list(range(2, 15)))
parser.add_argument("-u", "--umap_iters", type=int, help="the number of iterations to do with UMAP", default=30)

args = parser.parse_args()


data = None         # source of data for building clusters
pdata = None        # source of data for predicting cluster labels
if args.data is None:
    data = load_data()
else:
    data = TrackTBIFile(args.data)

if args.pdata is None:
    pdata = load_data()
else:
    pdata = TrackTBIFile(args.pdata)


fkwargs = dict()

cluster_sizes = args.cluster_sizes
n_iters = args.iterations

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seed = args.seed
if seed == None:
    seed = int(time.time())

if size > 1:
    seed = comm.bcast(seed, root=0)
    seed += rank
    fkwargs['driver'] = 'mpio'
    fkwargs['comm'] = comm

start, portion = get_start_portion(rank, size, n_iters)

logger = get_logger("bootstrap_umap_clustering", comm=comm)

labels, preds, rlabels, rpreds = subsample_umap_clustering(pdata.biomarkers, data.outcomes, cluster_sizes,
                                                           portion, args.fraction,
                                                           n_umap_iters=args.umap_iters, logger=logger)

shape = (n_iters, labels.shape[1], len(cluster_sizes))

if size > 1:
    comm.barrier()

if rank == 0:
    logger.info('writting results')

fkwargs['mode'] = 'w'
f = h5py.File(args.output, **fkwargs)

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
dset = f.create_dataset('umap_iters', data=args.umap_iters)
dset.attrs['description'] = "The number of UMAP iterations used for calculating the distance matrix"
dset = f.create_dataset('fraction', data=args.fraction)
dset.attrs['description'] = "the fraction to subsample at each iteration"

f.close()

if rank == 0:
    logger.info('done')
