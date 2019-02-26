import numpy as np
import scipy
from activ import load_data
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from activ.data_normalization import data_normalization
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PyUoI.UoINMF import UoINMF

rawdata = load_data(uoinmf=False)
raw_bm = rawdata.biomarkers
raw_oc = rawdata.outcomes

knum = 5
kf = StratifiedKFold(n_splits=knum)

transformed_raw_oc = raw_oc - np.min(raw_oc, axis=0)
oc_input = transformed_raw_oc

eps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
minsamples = np.arange(20,80,5)


error_mat = np.zeros((len(eps),len(minsamples),5))
k_mat = np.zeros((len(eps), len(minsamples), 5))
for cv, (train_index, test_index) in enumerate(kf.split(oc_input, raw_oc[:,32])):
    train, test = oc_input[train_index], oc_input[test_index]
    train_oc = train
    test_oc = test
    for ii,e in enumerate(eps):
        for jj,m in enumerate(minsamples):
    	    print('eps {} minsamples {}'.format(e,m))
            db = DBSCAN(eps=e, min_samples=m)
            uoinmf = UoINMF(ranks=list(range(2,20)),dbscan=db)
            decomp = uoinmf.fit(train_oc)
            H = decomp.components_
            W = decomp.transform(test_oc, reconstruction_err=True)
            error = np.linalg.norm(test_oc-W@H)
            error_mat[ii,jj,cv] = error
            k_mat[ii,jj,cv] = H.shape[0]

with h5py.File('/home/ahyeon96/results/uoinmf_oc_reconerr.h5', 'a') as f:
    f.create_dataset('error_mat', data=error_mat)
    f.create_dataset('k_mat', data=k_mat)


