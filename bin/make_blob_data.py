import argparse
import numpy as np
import h5py
import time

from sklearn.datasets import make_blobs

from activ.readfile import TrackTBIFile

def seed(x):
    if len(x) == 0:
        return int(round(time.time() * 1000)) % (2**32 - 1)
    try:
        return int(x)
    except Exception as e:
        raise argparse.ArgumentTypeError(e.args[0])

parser = argparse.ArgumentParser()
parser.add_argument('outfile', type=str, help='the output HDF5 file')
parser.add_argument('-n', '--n_samples', type=int, help='number of samples', default=100)
parser.add_argument('-p', '--n_bm_features', type=int, help='number of features', default=2)
parser.add_argument('-q', '--n_oc_features', type=int, help='number of features', default=2)
parser.add_argument('-b', '--n_blobs', type=int, help='number of blobs', default=3)
parser.add_argument('-s', '--seed', type=seed, help='random seed. default is based on clock', default='')


args = parser.parse_args()

rs = np.random.RandomState(args.seed)

X, labels = make_blobs(n_samples=args.n_samples,
                       n_features=args.n_bm_features,
                       centers=args.n_blobs,
                       random_state=rs)

beta = rs.normal(0, 1, (args.n_bm_features, args.n_oc_features))
Y = X @ beta

with h5py.File(args.outfile, 'w') as f:
    TrackTBIFile.write(f, X, Y)
    f.create_dataset('labels', data=labels)
    f.create_dataset('beta', data=beta)
    f.attrs['seeds'] = args.seed


