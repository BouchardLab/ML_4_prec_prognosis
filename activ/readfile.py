import h5py


class ReadFile(object):

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename) as f:
            self.data_bm = f['data_matrix_subset_biomarker'][:]
            self.data_oc = f['data_matrix_subset_outcome'][:]
            self.feature_bm = f['feature_name_subset_biomarker'][:]
            self.feature_oc = f['feature_name_subset_outcome'][:]
            self.id = f['patient_id'][:]
