import argparse
import time

import h5py
from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import scale


from activ import TrackTBIFile
from activ.utils import check_seed
from activ.cca.alscca import TALSCCA
from activ.cca.utils import compute_explained_variance


parser = argparse.ArgumentParser()
parser.add_argument('tbifile', type=str, help='the TrackTBIFile')
parser.add_argument('output', type=str, help='the output HDF5 file')
parser.add_argument('n_perm', type=int, help='the number of permutations to do')
parser.add_argument('seed', type=check_seed, help='the seed to use')

args = parser.parse_args()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



# read inputs
tbifile = TrackTBIFile(args.tbifile)
X = scale(tbifile.biomarkers, with_std=False)
Y = scale(tbifile.outcomes, with_std=False)
q = min(X.shape[1], Y.shape[1])

# set up other inputs
X_orig = X
rng = np.random.default_rng(args.seed)
params = {'l1_ratio_y': 1.0, 'l1_ratio_x': 1.0, 'alpha_y': 0.5, 'alpha_x': 0.5, 'n_components': 177, 'max_iter': 4000}


# set up output file
f = h5py.File(args.output, comm=comm)
f.attrs['seed'] = args.seed
perm_dset = f.create_dataset('permutation_indices', dtype=int, shape=(args.n_perm, X.shape[0]))
X_weights_dset = f.create_dataset('X_weights', dtype=float, shape=(args.n_perm, X.shape[1], q))
Y_weights_dset = f.create_dataset('Y_weights', dtype=float, shape=(args.n_perm, Y.shape[1], q))
expvar_dset = f.create_dataset('explained_variance', dtype=float, shape=(args.n_perm, q))



for i in range(rank, n_perms, size):
    idx = rng.permutation(np.arange(X_orig.shape[0]))
    perm_dset[i] = idx

    X = X_orig[idx]
    cca = TALSCCA(**params)
    with warnings.catch_warnings(record=False):
        warnings.simplefilter('ignore')
        _bm, _oc = cca.fit_transform(X, Y)

    X_weights_dset[i] = cca.x_weights_
    Y_weights_dset[i] = cca.y_weights_

    cv_X, cv_Y = talscca.transform(X, Y)
    exp_var, n_comp = compute_explained_variance(cv_X, cv_Y, Y, perc_var=1.0, method='cos')
    expvar_dset[i] = exp_var

f.close()
