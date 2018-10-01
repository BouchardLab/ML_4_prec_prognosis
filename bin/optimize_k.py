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



data = load_data()
data_oc = data_normalization(data.outcomes, 'positive')
train_oc, test_oc = train_test_split(data_oc)

def optimize_k_error(data, mpicomm=None):
   # rank = 0
   # if mpicomm:
   #     rank = mpicomm.Get_rank()
   #     iterations = range(rank, n_iters, mpicomm.Get_size())
   #     if rank != 0:
   #         seed = None
   #     seed = mpicomm.bcast(seed, root=0)
   #     seed += rank
   # _np.random.seed(seed)

    eps = np.arange(0.1,1.1,0.1)
    minsamples = np.arange(5,55,5)
    kf = KFold(n_splits=10)
    error_mat = np.zeros((10,10,10))
    for k, (train_index, test_index) in enumerate(kf.split(data_oc)):
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
                error_mat[ii,jj,k] = errorerror_mat = optimize_k_error(data_oc)
    return error_mat

error_mat = optimize_k_error(data_oc)
plt.matshow(error_mat, cmap='gray_r')
