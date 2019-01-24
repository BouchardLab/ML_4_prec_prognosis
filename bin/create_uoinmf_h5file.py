import h5py
from activ.readfile import TrackTBIFile
def read_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    print(filename, data.T.shape)
    return data.T

bm_W1 = '/Users/ahyeon/Desktop/TBIanalysis/data_matrix_subset_biomarker.h5'
bm_H1 = '/Users/ahyeon/Desktop/TBIanalysis/data_matrix_subset_biomarker_H1.h5'
oc_W1 = '/Users/ahyeon/Desktop/TBIanalysis/data_matrix_subset_outcome.h5'
oc_H1 = '/Users/ahyeon/Desktop/TBIanalysis/data_matrix_subset_outcome_H1.h5'

bm_w1 = read_h5(bm_W1)
bm_h1 = read_h5(bm_H1)
oc_w1 = read_h5(oc_W1)
oc_h1 = read_h5(oc_H1)

filename = '/Users/ahyeon/Desktop/TBIanalysis/data/data.h5'
with h5py.File(filename, 'r') as f:
    data_bm = f['data_matrix_subset_biomarker'][:].T
    data_oc = f['data_matrix_subset_outcome'][:].T
    feature_bm = [s.decode('utf-8') for s in f['feature_name_subset_biomarker'][:]]
    feature_oc = [s.decode('utf-8') for s in f['feature_name_subset_outcome'][:]]
    pid = [s.decode('utf-8') for s in f['patient_id'][:]]

data_uoinmf_filename = '/Users/ahyeon/Desktop/activ/activ/data_uoinmf.h5'
TrackTBIFile.write(data_uoinmf_filename, bm_w1, oc_w1, patient_ids=pid)

with h5py.File(data_uoinmf_filename, 'a') as f:
    f.create_dataset('bm_h1', data=bm_h1)
    f.create_dataset('oc_h1', data=oc_h1)

newfile = h5py.File(data_uoinmf_filename, 'r')
print(list(newfile.keys()))

