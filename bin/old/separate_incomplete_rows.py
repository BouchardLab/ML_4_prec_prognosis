import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from h5py import File, special_dtype

import activ.readfile as rf

def trim_variables(X, min_sample_occ=1.0, min_feature_occ=0.7):
    occ = np.ones(X.shape, dtype=int)
    occ[np.where(pd.isnull(X))] = 0
    col_perc_occ = np.sum(occ, 0)/X.shape[0]
    col_keep = np.where(col_perc_occ >= min_feature_occ)
    X_trim = X.iloc[:,col_keep[0]]
    occ = np.ones(X_trim.shape, dtype=int)
    occ[np.where(pd.isnull(X_trim))] = 0
    row_perc_occ = np.sum(occ, 1)/X_trim.shape[1]
    row_keep = np.where(row_perc_occ >= min_sample_occ)
    X_trim = X_trim.iloc[row_keep[0],]
    return (X_trim, col_keep[0])

def cleanup(df):
    # expand categorical features into binary features
    vec = DictVectorizer()
    onehot = vec.fit_transform(df.T.to_dict().values()).toarray()
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed = imputer.fit_transform(onehot)
    colmask = np.std(imputed, axis=0) > 0.0
    imputed = imputed[:, colmask]
    colnames = [col for i, col in enumerate(vec.get_feature_names()) if colmask[i]]
    return pd.DataFrame(imputed, columns=colnames)


preproc = rf.load_preprocessed(trim_col=True, impute_missing=True)

patient_ids = preproc.index
oc_mask = rf.load_outcome_mask(trim_col=True)
n_oc = np.sum(oc_mask)
n_bm = len(oc_mask) - n_oc
print(n_bm, "biomarkers", n_oc, "outcomes", preproc.shape[0], "samples")
print(preproc.shape[0], "samples with", preproc.shape[1], "features", "(%d biomarkers and %d outcomes)" % (n_bm, n_oc))

oc = preproc.iloc[:,oc_mask]
bm = preproc.iloc[:,np.logical_not(oc_mask)]

oc, ocbin = rf.encode(oc, get_binvars=True)
bm, bmbin = rf.encode(bm, get_binvars=True)

trimmed_oc, inv_oc = rf.filter_invariant(oc, get_inv_col=True)
trimmed_bm, inv_bm = rf.filter_invariant(bm, get_inv_col=True)

#print(ocbin)
#print(bmbin)
#print(inv_oc)
#print(inv_bm)

sample_mask = rf.load_rowmask()

strvl = special_dtype(vlen=bytes)
strtype = np.dtype('S7')

def tobytes(it):
    return [bytes(x, 'utf-8') for x in it]

bm = trimmed_bm
oc = trimmed_oc
with File('new_data.h5', 'w') as f:
    dset_bm = f.create_dataset("data_matrix_subset_biomarker", data=bm.iloc[sample_mask,:].values.T)
    dset_oc = f.create_dataset("data_matrix_subset_outcome", data=oc.iloc[sample_mask,:].values.T)
    print("After clean up, %s samples with %s biomarkers and %s outcomes" % (dset_bm.shape[1], dset_bm.shape[0], dset_oc.shape[0]))
    pdset = f.create_dataset("patient_id", data=tobytes(patient_ids[sample_mask]), dtype=strtype)
    f.create_dataset("feature_name_subset_biomarker", data=tobytes(bm.columns), dtype=strvl)
    f.create_dataset("feature_name_subset_outcomes", data=tobytes(oc.columns), dtype=strvl)
    disc = f.create_group('discarded')
    discarded_mask = np.logical_not(sample_mask)
    disc.create_dataset("data_matrix_subset_biomarker", data=bm.iloc[discarded_mask,:].values.T)
    disc.create_dataset("data_matrix_subset_outcome", data=oc.iloc[discarded_mask,:].values.T)
    disc.create_dataset("patient_id", data=tobytes(patient_ids[discarded_mask]), dtype=strvl)
    #tfm = f.create_group('transformation')
    #strmat = [np.array([bytes(k, 'utf-8')] + tobytes(v)) for k, v in bmbin.items()]
    #dset = tfm.create_dataset("binencode_biomarker", dtype=strvl, shape=(len(strmat), 3))
    #for i, row in enumerate(strmat):
    #    print(row)
    #    dset[i] = row
