import h5py


class ReadFile(object):

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename) as f:
            self.data_matrix_subset_biomarker = f['data_matrix_subset_biomarker'][:]
            self.data_matrix_subset_outcome = f['data_matrix_subset_outcome'][:]
            self.feature_name_subset_biomarker = f['feature_name_subset_biomarker'][:]
            self.feature_name_subset_outcome = f['feature_name_subset_outcome'][:]
            self.patient_id = f['patient_id'][:]
