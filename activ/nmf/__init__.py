import numpy as np
from pyuoi.decomposition import UoI_NMF_Base
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans


def bases_factor_order(bases):
    return np.argsort(np.max(np.asarray(bases), axis=1))[::-1]


def run_nmf(mat, n_components, alpha, l1_ratio, seed=None):
    nmf = NMF(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, solver='mu', beta_loss='kullback-leibler')
    nmf = UoI_NMF_Base(n_boots=20, ranks=[n_components], nmf=nmf, cluster=KMeans(n_clusters=n_components), cons_meth=np.median, random_state=seed+1)
    nmf.fit(mat)
    weights = nmf.transform(mat)
    bases = nmf.components_
    row_order = bases_factor_order(bases)
    return nmf, weights[:, row_order], bases[row_order]
