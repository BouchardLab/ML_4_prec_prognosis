from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist


def decomp(data, method_obj):
    return method_obj.fit_transform(data)


def eval(data, method_obj, eval_method):
    decomposed = decomp(data, method_obj)
    return eval_method.eval(data, decomposed)


def sl_method(X_train, X_test, y_train, y_test, *args):
    # method_obj: UoILasso, UoIRandomForest, iRF
    scores = {}
    for sl_method_obj in args:
        sl_method_obj.fit(X_train, y_train)
        prediction = sl_method_obj.predict(X_test)
        accuracy = method.score(y_test, prediction)
        scores[sl_method_obj] = accuracy
    best_method = max(scores.items(), key = lambda x: x[1])[0]
    return best_method


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


def cluster(cluster_method, nclust_range, X, method='single', metric='euclidean'):
    dist = pdist(X, metric=metric)
    linkmat = linkage(dist, method=method)
    return cut_tree(linkmat, nclust_range)


def score_clusters(biomarkers, cluster_ids, classifier, train_frac):
    bm_train, bm_test, ids_train, ids_test = train_test_split(biomarkers, cluster_ids, train_size=train_frac)
    nclust_range = len(cluster_ids[0])
    ret = np.zeros(nclust_range, dtype=np.float)
    for i in range(nclust_range):
        classifier.fit(bm_train, ids_train[:,i])
        pred = classifier.predict(bm_test)
        ret[i] = accuracy_score(ids_test[:,i], pred)
    return ret
