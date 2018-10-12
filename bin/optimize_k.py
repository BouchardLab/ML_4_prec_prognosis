import matplotlib
matplotlib.use('Agg')
from activ import load_data, data_normalization
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PyUoI.UoINMF import UoINMF
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold
from mpi4py import MPI
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('eps', type=list, help='a list of eps values', default=list(range(0.3,0.6,0.1)))
parser.add_argument('minsamples', type=int_list, help='a list of minsample values', default=list(range(5,55,5)))
parser.add_argument('kfold', type=int, help='number of kfolds')

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()

data = load_data()
data_oc = data_normalization(data.outcomes, 'positive')

kf = KFold(n_splits=args.kfold)
indices = range(rank, args.kfold, size)

error_mat = np.zeros((len(args.eps), len(args.minsamples), args.kfold))
k_mat = np.zeros((len(args.eps), len(args.minsamples), args.kfold))

for cv, (train_index, test_index) in zip(indices, kf.split(data_oc)):
    train, test = data_oc[train_index], data_oc[test_index]
    train_oc = train
    test_oc = test
    for ii,e in enumerate(args.eps):
        for jj,m in enumerate(args.minsamples):
            db = DBSCAN(eps=e, min_samples=m)
            uoinmf = UoINMF(ranks=list(range(2,20)),dbscan=db)
            decomp = uoinmf.fit(train_oc)
            H = decomp.components_
            k = H.shape[0]
            W = decomp.transform(test_oc, reconstruction_err=True)
            error = np.linalg.norm(test_oc-W@H)
            error_mat[ii,jj,cv] = error
            k_mat[ii,jj,cv] = k

f = h5py.File('optimize_k_mat.h5', 'w', driver="mpio", comm=MPI.COMM_WORLD)
f.create_dataset('error', data=error_mat)
f.create_dataset('k', data=k_mat)
f.close()
