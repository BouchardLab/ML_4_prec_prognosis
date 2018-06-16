from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py



def decomp(data, method_obj):
    return method_obj.fit_transform(data)

def eval(data, method_obj, eval_method):
    decomposed = decomp(data, method_obj)
    return eval_method.eval(data, decomposed)

def sl_method(X_train, y_train, sl_method_obj):
    # method_obj: UoILasso, UoIRandomForest, iRF 
    return sl_method_obj.fit(X_train, y_train)


    # prediction = sl_method_obj.predict(X_test)
    # accuracy = score_func(y_test, prediction)
    # return accuracy

def best_decomp_method(data, *args):
    scores = {}
    for method in args:
        decomp_data = decomp(data, method)
        eval_score = method.score(data, decomp_data)
        # eval_score = eval(data, method, eval_method)
        scores[method] = eval_score
    best_method = max(scores.items(), key = lambda x: x[1])[0]
    return best_method

def build_predictive_model(biomarkers, outcomes, bio_method, out_method, sl_method_obj):
    biomarkers_dec = decomp(biomarkers, bio_method)
    outcomes_dec = decomp(outcomes, out_method)
    score = sl_method(biomarkers_dec, outcomes_dec, sl_method_obj)
    return score

if __name__ == '__main__':

    # biomarker_dec: NMF, ICA, DL (ditionary learning)
    # outcome_dec: PCA, FA (factor analysis), MDS (multidim scaling), UMAP
    
    # range(1,11)
    pca = PCA(n_components = 2)
    fa = FactorAnalysis()
    mds = MDS(n_components = 2)
    lr = LogisticRegression()
        
    # biomarkers = [0,1,2,3,4,5]
    # outcomes = [1,2,3,4,5,6]
    # mnist = datasets.load_digits()
    # mnist_X = mnist.data
    # mnist_y = mnist.target

    # mnist_X = biomarker_dec(mnist_X, pca)
    # mnist_y = outcome_dec(mnist_y, pca)

    # X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, test_size = 0.2)
    
    # accuracy = sl_method(X_train, X_test, y_train, y_test, lr, score_func = accuracy_score)
    # print('accuracy: {}'.format(accuracy))

    data = '/Users/ahyeon/Desktop/TBIanalysis/data/data.h5'
    f = h5py.File(data, 'r')
    data_matrix_subset_biomarker = f['data_matrix_subset_biomarker']
    data_matrix_subset_outcome = f['data_matrix_subset_outcome']
    feature_name_subset_biomarker = f['feature_name_subset_biomarker']
    feature_name_subset_outcome = f['feature_name_subset_outcome']
    patient_id = f['patient_id']

    method = best_decomp_method(data_matrix_subset_biomarker, pca, fa)
    print(method)

  
    

