from activ.pipeline import best_decomp_method

if __name__ == '__main__':

    # biomarker_dec: NMF, ICA, DL (dictionary learning)
    # outcome_dec: PCA, FA (factor analysis), MDS (multidim scaling), UMAP
    
    mds = MDS()
    lr = LogisticRegression()

    data = '/Users/ahyeon/Desktop/TBIanalysis/data/data.h5'
    f = h5py.File(data, 'r')
    data_matrix_subset_biomarker = f['data_matrix_subset_biomarker']
    data_matrix_subset_outcome = f['data_matrix_subset_outcome']
    feature_name_subset_biomarker = f['feature_name_subset_biomarker']
    feature_name_subset_outcome = f['feature_name_subset_outcome']
    patient_id = f['patient_id']
    for n in range(1,11):
        pca = PCA(n_components = n)
        fa = FactorAnalysis(n_components = n)
        method = best_decomp_method(data_matrix_subset_biomarker, pca, fa)
        print(method)