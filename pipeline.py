from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


def biomarker_dec(biomarkers, method_obj):
	# method_obj: NMF, ICA, DL (ditionary learning)
	return method_obj.fit_transform(biomarkers)

def outcome_dec(outcomes, method_obj):
	# method_obj: PCA, FA (factor analysis), MDS (multidim scaling), UMAP
	return method_obj.fit_transform(outcomes)

	# def outcome_dec_optimization()

def sl_method(X_train, y_train, sl_method_obj):
	# method_obj: UoILasso, UoIRandomForest, iRF 
	sl_method_obj.fit(X_train, y_train)
	# prediction = sl_method_obj.predict(X_test)
	# accuracy = score_func(y_test, prediction)
	# return accuracy

def ML_pipeline(biomarkers, outcomes, biomarker_dec_method, outcome_dec_meth, sl_method):
	bm_dec = biomarker_dec(biomarkers, bdm)
	oc_dec = outcome_dec(outcomes, ocm)

	bm_sl = sl_method(bm_dec, oc_dec, bslm)


	return bm_sl, oc_sls

if __name__ == '__main__':
	pca = PCA(n_components=2)
	lr = LogisticRegression()
	# mds = MDS(...)
	# rf = RandomForestRegressor(...)	
	biomarkers = [0,1,2,3,4,5]
	outcomes = [1,2,3,4,5,6]
	mnist = datasets.load_digits()
	mnist_X = mnist.data
	mnist_y = mnist.target

	mnist_X = biomarker_dec(mnist_X, pca)
	mnist_y = outcome_dec(mnist_y, pca)

	X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, test_size = 0.2)
	
	accuracy = sl_method(X_train, X_test, y_train, y_test, lr, score_func = accuracy_score)
	print('accuracy: {}'.format(accuracy))


