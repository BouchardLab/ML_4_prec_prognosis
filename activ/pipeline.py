from sklearn.decomposition import PCA, FactorAnalysis, NMF, FastICA, DictionaryLearning
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist
from umap import UMAP
import numpy as np
import numpy.matlib as npm

from .data_normalization import data_normalization


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


def cluster_range(X, n_clusters, method='single', metric='euclidean'):
    """
    Cluster data across a range of number of clusers using hierarchical
    clustering.

    Args:
        X (ndarray):             an n by p array of observations
        n_clusters (array_like): the number of clusters
        method (str):            the linkage method. See scipy.cluster.hierarchy.linkage
                                 for available methods
        metric (str):            the distance metric. See scipy.distance.pdist
                                 for available metrics

    Return:
        an n x len(n_clusters) array. cluster assignments for each observation
        across for each element in n_clusters
    """
    dist = pdist(X, metric=metric)
    linkmat = linkage(dist, method=method)
    return cut_tree(linkmat, n_clusters)


def score_clusters(X, cluster_ids, classifier=RFC(100), train_frac=0.8):
    """
    Score each clustering using the given classifier.

    Args:
        X (ndarray):                         an n by p array of observations
        cluster_ids (array_like):            the cluster assignments as returned by cluster_range
        classifier (sklearn classifier):     the classification method to use.
                                             default is RandomForestClassifier(n_estimators=100)
        train_frac (float):                  the fraction of the data to use for training
                                             default is 0.8

    Return:
        the predictive accuracy for each value of n_clusters as given to cluster_range
    """
    bm_train, bm_test, ids_train, ids_test = train_test_split(X, cluster_ids, train_size=train_frac)
    nclust_range = len(cluster_ids[0])
    ret = np.zeros(nclust_range, dtype=np.float)
    for i in range(nclust_range):
        classifier.fit(bm_train, ids_train[:,i])
        pred = classifier.predict(bm_test)
        ret[i] = accuracy_score(ids_test[:,i], pred)
    return ret


def score_clusters_cv(X, cluster_ids, classifier=RFC(100), cv=10):
    """
    Score each clustering using the given classifier.

    Args:
        X (ndarray):                         an n by p array of observations
        cluster_ids (array_like):            the cluster assignments as returned by cluster_range
        classifier (sklearn classifier):     the classification method to use.
                                             Default is RandomForestClassifier(n_estimators=100)
        cv (int):                            the number of folds to use. Default is 3.
                                             See sklearn.model_selection.cross_val_score for more
                                             details

    Return:
        the predictive accuracy (averaged across k folds of CV) for each value of n_clusters as
        given to cluster_range
    """
    nclust_range = len(cluster_ids[0])
    ret = np.zeros((nclust_range, cv), dtype=np.float)
    print(ret.shape)
    for i in range(nclust_range):
        ret[i,:] = cross_val_score(classifier, X, y=cluster_ids[:,i], cv=cv)
    return ret


def filter_outliers(X, num_outliers):
    """
    Remove outliers from a data matrix

    Args:
        X (ndarray):         an n by p array of observations
        num_outliers (int):  the number of outliers to discard

    Return:
        a tuple of the resulting filtered data, and the indices of the
        discarded data points

    """
    n = X.shape[0]
    mask = np.ones(n)
    means = np.mean(X, axis=0)
    discard = np.sort(np.argsort(np.sum((X - npm.repmat(means, n, 1))**2, axis=1))[n-num_outliers:])
    mask[discard] = 0
    filtered = X[mask==1]
    return filtered, discard

def pca(X, num_latent, norm='z-score'):
    """
    Run PCA (after normalizing) and return the first num_latent
    components

    Args:
        X (ndarray):       an n by p array of observations
        num_latent (int):  the number of principal components to keep
        norm (str):        the data normalization strategy. See data_normalization
                           for more details.


    Return:
        The transformed data

    """
    normed = data_normalization(X, norm)
    ret = PCA(n_components=num_latent).fit_transform(normed)
    return ret

def run_umap(X,n_components, n_neighbors=10, min_dist=0.1, metric='euclidean', norm='z-score', random_state=None):
    """
    Run UMAP (after normalizing) and return the transformed data

    Args:
        n_neighbors (int):      Somewhere in the range of 5 to 50. default is 10
                                See UMAP docs for more details.
        min_dist (float):       Somewhere in the range of 0.001 to 0.5. default is 0.1
                                See UMAP docs for more details.
        metric (str):           The distance metric to use. default is euclidean
                                See UMAP docs for all available metrics
    """
    normed = data_normalization(X, norm)
    ret = UMAP(n_components=n_components,
               n_neighbors=n_neighbors,
               min_dist=min_dist,
               metric=metric,
               random_state=random_state).fit_transform(X)
    return ret
