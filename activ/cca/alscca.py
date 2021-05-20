from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA
import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import enet_path, ridge_regression


def inprod(u, M, v=None):
    v = u if v is None else v
    return u.dot(M.dot(v.T))


def gs(A, M):
    """
    Gram-Schmidt with inner product for M
    """
    orig_A = A
    A = A.copy()
    A[:, 0] = A[:, 0] / np.sqrt(inprod(A[:, 0], M))
    for i in range(1, A.shape[1]):
        Ai = A[:, i]
        for j in range(0, i):
            Aj = A[:, j]
            t = inprod(Ai, M, Aj)
            Ai = Ai - t * Aj
        norm = np.sqrt(inprod(Ai, M))
        if norm == 0:
            A[:, i] = Ai
        else:
            A[:, i] = Ai / norm
    return A


def cdsolve(X, Y, init, reg, random_state, l1_ratio=1.0, max_iter=1000, tol=0.1, selection='cyclic'):
    """
    Returns:
        coef (array)        : shape is (n_predictors, n_responses)
    """
    try:

        if l1_ratio == 0.0:
            ret = ridge_regression(X, Y, reg, random_state=random_state, tol=tol,
                             return_n_iter=False, max_iter=max_iter)
            ret = ret.T
        else:
            _, ret, this_dual_gap = \
                enet_path(X, Y, coef_init=init.T, alphas=[reg],
                          random_state=random_state, max_iter=max_iter,
                          selection=selection, precompute='auto',
                          check_input=True, return_n_iters=False,
                          tol=tol,
                          l1_ratio=l1_ratio, eps=None, n_alphas=None)
            ret = ret.squeeze().T
    except ValueError as e:
        raise e

    return ret


def tals_cca(X, Y, k, max_iters=1000, tol=0.0001, random_state=None,
             alpha_x=0.001, alpha_y=0.001,
             l1_ratio_x=1.0, l1_ratio_y=1.0,
             return_cov=False, return_delta=False):
    random_state = check_random_state(random_state)
    n = len(X)
    p = X.shape[1]
    q = Y.shape[1]
    Cxx = X.T.dot(X)/n + alpha_x*np.identity(p)    # regularize for
    Cyy = Y.T.dot(Y)/n + alpha_y*np.identity(q)    # stability
    Cxy = X.T.dot(Y)/n
    H_0 = random_state.standard_normal((p, k))
    S_0 = random_state.standard_normal((q, k))
    H_t1 = gs(H_0, Cxx)
    S_t1 = gs(S_0, Cyy)

    H_t = None
    S_t = None

    n_iters = 0
    deltas = list()
    for _ in range(max_iters):
        H_init = H_t1.dot(LA.pinv(inprod(H_t1.T, Cxx)).dot(inprod(H_t1.T, Cxy, S_t1.T)))
        # solve for H_t, initialized at H_init, use S_t1
        H_t = cdsolve(X, Y.dot(S_t1), H_init, alpha_x/2, random_state, l1_ratio=l1_ratio_x)
        H_t = gs(H_t, Cxx)
        S_init = S_t1.dot(LA.pinv(inprod(S_t1.T, Cyy)).dot(inprod(S_t1.T, Cxy.T, H_t.T)))
        # solve for S_t, initialized at S_init, use H_t
        S_t = cdsolve(Y, X.dot(H_t), S_init, alpha_y/2, random_state, l1_ratio=l1_ratio_y)
        S_t = gs(S_t, Cyy)
        n_iters += 1
        delta = np.abs(np.concatenate([(H_t - H_t1), (S_t - S_t1)])).mean()
        deltas.append(delta)
        if delta < tol:
            break
        H_t1 = H_t
        S_t1 = S_t

    ret = [H_t, S_t, n_iters]
    if return_cov:
        ret.append(Cxx)
        ret.append(Cyy)
        ret.append(Cxy)

    if return_delta:
        ret.append(np.array(deltas))
    return tuple(ret)


class TALSCCA(BaseEstimator):
    """
    Alternating Least Squares Canonical Correlation Analysis
    """

    def __init__(self, random_state=None, scale=False,
                 n_components=1, max_iters=100, alpha_x=0.001,
                 alpha_y=0.001, l1_ratio_x=1.0, l1_ratio_y=1.0, tol=0.001):
        """
        Args:
            n_components:               the number of canonical variates to compute
            max_iter:                   maximum number of alternating least-squares
                                        iterations
            alpha_x:                    regularization for regressing Y onto X
            alpha_y:                    regularization for regressing X onto Y
            scale:                      standardize X and Y matrices before fitting
        """

        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.l1_ratio_x = l1_ratio_x
        self.l1_ratio_y = l1_ratio_y
        self.x_scale = None
        self.y_scale = None
        if scale:
            self.x_scale = StandardScaler()
            self.y_scale = StandardScaler()

    def fit(self, X, Y):
        if self.x_scale is not None:
            X = self.x_scale.fit_transform(X)
            Y = self.x_scale.fit_transform(Y)
        ret = tals_cca(X, Y, self.n_components, max_iters=self.max_iters, tol=self.tol,
                       alpha_x=self.alpha_x, alpha_y=self.alpha_y, return_cov=True,
                       l1_ratio_x=self.l1_ratio_x, l1_ratio_y=self.l1_ratio_y,
                       random_state=self.random_state, return_delta=True)

        H, S, self.n_iters_ = ret[0], ret[1], ret[2]
        self.x_weights_, self.y_weights_ = H, S
        self.X_cov_ = ret[3]
        self.Y_cov_ = ret[4]
        self.XY_cov_ = ret[5]
        self.deltas_ = ret[6]
        return self

    def transform(self, X, Y):
        if getattr(self, 'x_weights_', None) is None:
            raise ValueError("TALSCCA is not fit")
        if self.x_scale is not None:
            X = self.x_scale.transform(X)
            Y = self.y_scale.transform(Y)
        return X.dot(self.x_weights_), Y.dot(self.y_weights_)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X,Y)

class ALSCCA(BaseEstimator):
    """
    Alternating Least Squares Canonical Correlation Analysis
    """


    def __init__(self, X_lm=None, Y_lm = None, random_state=None, n_components=1, max_iters=100, alpha=0.1):
        """
        Args:
            X_lm:                     Linear model for regressing X onto Y
            Y_lm:                     Linear model for regressing Y onto X
            max_iter:                 max_iter for default Lasso object
            alpha:                    alpha for default Lasso object
        """
        if X_lm is None:
            X_lm = Lasso(alpha=alpha, max_iter=2000)
        if Y_lm is None:
            Y_lm = Lasso(alpha=alpha, max_iter=2000)

        self.X_lm = X_lm
        self.Y_lm = Y_lm

        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.max_iters = max_iters

    def fit(self, X, Y):

        theta = normalize(self.random_state.rand(Y.shape[1], self.n_components), norm='l2', axis=0)

        beta = self.X_lm.fit(X, Y@theta).coef_.T

        n_iters = 10
        beta_d = np.zeros(n_iters)
        theta_d = np.zeros(n_iters)

        for i in range(n_iters):
            theta_new = self.Y_lm.fit(Y, X@beta).coef_.T
            beta_new = self.X_lm.fit(X, Y@theta).coef_.T
            beta_d[i] = np.linalg.norm(beta_new - beta)
            theta_d[i] = np.linalg.norm(theta_new - theta)
            beta = beta_new
            theta = theta_new
        self.beta = beta
        self.theta = theta
        self.beta_d = beta_d
        self.theta_d = theta_d
        return self

    def transform(self, X, Y):
        z = X @ self.beta
        s = Y @ self.theta
        return z, s


from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree
import scipy.stats as sps
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import ParameterGrid

class UoI_sCCA_Base(BaseEstimator):


    def __init__(self, n_components=2, metric='jaccard', binary_cluster=True, n_boots=20, intersect_fraction=1.0,
                       subsample_fraction=0.9, random_state=None, score_function='bic'):
        """
        Args:
            binary_cluster (bool)           : cluster samples based on absence/presence of weights. default=True
            metric (string)                 : the distance metric to use. If *binary_cluster* is True, use 'jaccard'.
                                              If *binary_cluster* is False, use 'euclidean'
        """
        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.n_boots = n_boots
        self.subsample_fraction = subsample_fraction
        self.est_cca = CCA(n_components=1, scale=False)
        self.sel_cca = TALSCCA()
        self.intersect_fraction = intersect_fraction
        if score_function == 'bic':
            self.score_function = self.bic
        elif score_function == 'pearsonr':
            self.score_function = self.pearsonr
        else:
            msg = "Unrecognized score_function: '%s' -- please choose from 'bic' or 'pearsonr'" % score_function
            raise ValueError(msg)
        self.binary_cluster = binary_cluster
        self.metric = metric
        if metric is None:
            if self.binary_cluster:
                self.metric = 'jaccard'
            else:
                self.metric = 'euclidean'

        self.params = ParameterGrid({'alpha_x': np.logspace(-5, -1, 5),
                                     'alpha_y': np.logspace(-5, -1, 5),
                                     'n_components': [n_components],
                                     'random_state': [self.random_state],
                                     'max_iters': [1000]})

        self.params = ParameterGrid({'alpha_x': np.logspace(-2, -1, 2),
                                     'alpha_y': np.logspace(-2, -1, 2),
                                     'n_components': [n_components],
                                     'random_state': [self.random_state],
                                     'max_iters': [1000]})


    @classmethod
    def intersect(cls, samples, axis=0, perc=1.0):
        return (np.mean(samples, axis=axis) > perc).astype(int)

    @classmethod
    def ll(cls, x, y):
        """
        The log likelihood of two canonical variates
        """
        return np.log(sps.norm().pdf(x-y)).sum()

    @classmethod
    def pearsonr(cls, x, y, parameters):
        return sps.pearsonr(x, y)[0]

    @classmethod
    def bic(cls, x, y, parameters):
        """
        Return the negative BIC. Negate the BIC to be consistent
        with other scoring measures.
        """
        k = np.sum(parameters != 0)
        L = cls.ll(x, y)
        n = len(x)
        return  2 * L - k * np.log(n)

    def fit(self, X, Y):
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        k = self.n_components
        n_params = len(self.params)

        weights_samples = np.zeros((len(self.params), self.n_boots, k, (p + q)))

        idx = np.arange(n)
        boot_samples = np.array([self.random_state.permutation(n) for i in range(self.n_boots)])
        rep_idx = int(self.subsample_fraction * n)

        #################
        # BEGIN selection
        #################
        print("selection")
        for param_j in range(n_params):
            for boot_i in range(self.n_boots):
                idx = boot_samples[boot_i, :rep_idx]
                X_rep, Y_rep = X[idx], Y[idx]
                print(self.params[param_j])
                self.sel_cca.set_params(**self.params[param_j])
                self.sel_cca.fit(X_rep, Y_rep)
                weights_samples[param_j, boot_i, :, :p] = self.sel_cca.x_weights_.T
                weights_samples[param_j, boot_i, :, p:] = self.sel_cca.y_weights_.T

        weights_samples = weights_samples.reshape(len(self.params), self.n_boots*k, p + q)
        models = np.zeros((n_params, k, p + q), dtype=bool)
        for param_j in range(n_params):
            # Cluster occupancy vectors
            samples = (weights_samples[param_j] != 0).astype(int)
            labels = cut_tree(linkage(pdist(samples, metric=self.metric), method='ward'), n_clusters=k)[:,0]   # This is the intersecting operation
            for label in np.unique(labels):   # labels <=> components
                models[param_j, label, :] = np.ceil(np.median(samples[labels==label], axis=0))

        self.weight_samples_ = weights_samples
        ###############
        # END selection
        ###############

        ##################
        # BEGIN estimation
        ##################
        print("estimation")
        scores = np.zeros((self.n_boots, n_params))
        estimates = np.zeros((self.n_boots, n_params, k*(p + q)))
        for model_i in range(n_params):
            for boot_i in range(self.n_boots):
                train_idx = boot_samples[boot_i, :rep_idx]
                test_idx = boot_samples[boot_i, rep_idx:]
                X_weight = np.zeros((k, p))
                Y_weight = np.zeros((k, q))
                X_pred = np.zeros((k*n, 1))
                Y_pred = np.zeros((k*n, 1))
                s = 0
                e = 0
                for component_i in range(k):
                    model = models[model_i, component_i]
                    X_mask, Y_mask = np.array_split(model, [p])
                    if np.logical_not(X_mask).all() or np.logical_not(Y_mask).all():
                        X_weight[component_i, X_mask] = 0
                        Y_weight[component_i, Y_mask] = 0
                        continue
                    X_train = X[train_idx][:, X_mask]
                    Y_train = Y[train_idx][:, Y_mask]
                    X_test = X[test_idx][:, X_mask]
                    Y_test = Y[test_idx][:, Y_mask]
                    self.est_cca.fit(X_train, Y_train)
                    X_weight[component_i, X_mask] = self.est_cca.x_weights_.flatten()
                    Y_weight[component_i, Y_mask] = self.est_cca.y_weights_.flatten()
                    e += X_test.shape[0]
                    _x, _y = self.est_cca.transform(X_test, Y_test)
                    if len(_x.shape) == 1: _x = _x.reshape(-1, 1)
                    if len(_y.shape) == 1: _y = _y.reshape(-1, 1)
                    X_pred[s:e], Y_pred[s:e] = _x, _y # self.est_cca.transform(X_test, Y_test)
                    s = e
                estimates[boot_i, model_i, :k*p] = X_weight.flatten()
                estimates[boot_i, model_i, k*p:] = Y_weight.flatten()
                scores[boot_i, model_i] = self.score_function(X_pred.flatten(), Y_pred.flatten(), estimates[boot_i, model_i])

        # calculate the best model for each bootstrap
        model_max_idx = np.argmax(scores, axis=1)
        best_estimates = estimates[np.arange(self.n_boots), model_max_idx, :]

        # aggregate across bootstraps
        weights = np.median(best_estimates, axis=0)

        self.x_weights_ = weights[:k*p].reshape(k, p).T
        self.y_weights_ = weights[k*p:].reshape(k, q).T
        ################
        # END estimation
        ################

        return self

    def transform(self, X, Y):
        if getattr(self, 'x_weights_', None) is None:
            raise ValueError("TALSCCA is not fit")
        return X.dot(self.x_weights_), Y.dot(self.y_weights_)
