import matplotlib
matplotlib.use('Agg')
from activ import load_data, data_normalization
import matplotlib.pyplot as plt
import numpy as np
import scipy
from activ import load_data
from PyUoI.UoINMF import UoINMF
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from activ.data_normalization import data_normalization
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()

data = load_data()
data_oc = data_normalization(data.outcomes, 'positive')

def optimize_k_error(data, mpicomm=None):
    eps = np.arange(0.1,1.1,0.1)
    minsamples = np.arange(5,55,5)
    n_splits = 10
    kf = KFold(n_splits=10)
    indices = range(rank, n_splits, size)
    error_mat = np.zeros((10,10,10))

    for k, (train_index, test_index) in zip(indices, kf.split(data_oc)):
        train, test = data_oc[train_index], data_oc[test_index]
        train_oc = train
        test_oc = test

        for ii,e in enumerate(eps):

            for jj,m in enumerate(minsamples):
                db = DBSCAN(eps=e, min_samples=m)
                uoinmf = UoINMF(ranks=list(range(2,20)),dbscan=db)
                decomp = uoinmf.fit(train_oc)
                H = decomp.components_
                W = decomp.transform(test_oc, reconstruction_err=True)
                error = np.linalg.norm(test_oc-W@H)
                error_mat[ii,jj,k] = error_mat = optimize_k_error(data_oc)
    return error_mat

error_mat = optimize_k_error(data_oc)

h5f = h5py.File('optimize_k_mat.h5', 'w', driver="mpio", comm=MPI.COMM_WORLD)
h5f.create_dataset('mat', data=error_mat)
h5f.close()

plt.matshow(error_mat, cmap='gray_r')
